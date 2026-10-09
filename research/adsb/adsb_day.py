# SPDX-License-Identifier: GPL-3.0-only
"""One adsb.lol globe_history day -> ground points and departure ground segments at the 10 PRC airports.

Usage (on the Mac, <mac-workdir>):
  .venv/bin/python adsb_day.py --day 2025-01-15 [--manifest manifest.tsv] [--out out] [--raw raw] [--keep-raw]
                               [--stream] [--no-download]

Steps
  1. download the day's split tar parts (curl, resumable) into raw/<day>/, timing each part
     (--stream skips the disk and parses curl's stdout directly)
  2. read the parts as one tar stream (nothing is extracted to disk), gunzip each traces/*/trace_full_*.json
     in memory, keep a trace only if a byte regex finds a "ground" point inside one of the airport boxes
  3. for kept traces: keep points within R_GROUND km of an airport on the ground, or within R_AIR km below
     elevation + ALT_AIR ft in the air, with callsign forward-filled from the aircraft dicts
  4. per (aircraft, airport): find takeoffs (ground -> airborne, confirmed by a climb) and describe the
     ground segment that ends in each takeoff (first point, stand stop, first motion, stops, takeoff)
  5. write out/<day>/points.parquet, events.parquet, meta.json and delete raw/<day>/ (unless --keep-raw)

All times are unix seconds (UTC). Speeds are knots, distances metres. Single thread.
"""
import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

try:
    import resource
except ImportError:  # Windows
    resource = None

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from isal import igzip as _gz

    def gunzip(b):
        return _gz.decompress(b)
except Exception:  # pragma: no cover
    import gzip as _gz

    def gunzip(b):
        return _gz.decompress(b)

AIRPORTS = {  # aerodrome reference point lat, lon, elevation ft (AIP values, rounded)
    'EDDF': (50.0333, 8.5706, 364),
    'EDDM': (48.3538, 11.7861, 1487),
    'EGLL': (51.4706, -0.4619, 83),
    'EHAM': (52.3086, 4.7639, -11),
    'LEBL': (41.2971, 2.0785, 12),
    'LEMD': (40.4719, -3.5626, 2001),
    'LFPG': (49.0097, 2.5479, 392),
    'LIRF': (41.8003, 12.2389, 15),
    'LSZH': (47.4647, 8.5492, 1416),
    'LTFM': (41.2753, 28.7519, 325),
}
R_GROUND = 8.0      # km, ground points kept
R_AIR = 12.0        # km, airborne points kept (takeoff confirmation)
ALT_AIR = 5000.0    # ft above elevation, airborne points kept

# segment and stop parameters
STAT_KT = 1.0       # a point at or below this ground speed stands still (within 2.5*STOP_R of the stop anchor)
MOVE_KT = 2.0       # a point at or above this ground speed moves, however close to the anchor
STOP_R = 20.0       # m, points below MOVE_KT or without speed stand still within this radius of the anchor
GAP_CUT = 600.0     # s, a data gap at least this long cuts the segment unless the aircraft stood still at the
                    #    same place on both sides (parked with the transponder off)
TURN_STOP = 600.0   # s, after a landing the stand stop is the first stop at least this long
LONG_STOP = 600.0   # s, the last stop at least this long gives the alternative stand (tow, then park)
MID_STOP = 60.0     # s, stops at least this long between first motion and takeoff are counted
TO_CONFIRM = 300.0  # s, takeoff confirmation window
TO_GAP_MAX = 180.0  # s, ground->air gap above this is flagged (roll not observed)


def km_box(la, lo, r):
    dlat = r / 111.2
    dlon = r / (111.2 * math.cos(math.radians(la)))
    return la - dlat, la + dlat, lo - dlon, lo + dlon


def tenth_prefixes(lo, hi):
    """Decimal strings with one decimal that start every number in [lo, hi] (e.g. '50.0', '-0.4')."""
    out = []
    v = lo
    while v <= hi + 0.05:
        x = min(v, hi)
        a = math.floor(abs(x) * 10 + 1e-9) / 10
        s = ('-' if x < 0 else '') + f'{a:.1f}'
        if s not in out:
            out.append(s)
        v += 0.05
    return out


def build_prefilter():
    alts = []
    for la, lo, _ in AIRPORTS.values():
        a0, a1, o0, o1 = km_box(la, lo, R_GROUND + 1.0)
        lp = '|'.join(re.escape(p) for p in tenth_prefixes(a0, a1))
        op = '|'.join(re.escape(p) for p in tenth_prefixes(o0, o1))
        alts.append('(?:' + lp + ')[0-9]*,(?:' + op + ')[0-9]*')
    pat = r'\n\[[0-9.]+,(?:' + '|'.join(alts) + r'),"ground",'
    return re.compile(pat.encode())


AP_LIST = list(AIRPORTS)
AP_LAT = np.array([AIRPORTS[a][0] for a in AP_LIST])
AP_LON = np.array([AIRPORTS[a][1] for a in AP_LIST])
AP_EL = np.array([AIRPORTS[a][2] for a in AP_LIST], dtype=float)
AP_COS = np.cos(np.radians(AP_LAT))


def nearest_airport(lat, lon):
    """Index of nearest airport and distance in km for arrays of points."""
    dy = (lat[:, None] - AP_LAT[None, :]) * 111.2
    dx = (lon[:, None] - AP_LON[None, :]) * 111.2 * AP_COS[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    k = np.argmin(d, axis=1)
    return k, d[np.arange(len(k)), k]


class MultiFile(io.RawIOBase):
    """Several files read back to back as one stream."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.i = 0
        self.f = open(self.paths[0], 'rb')

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            n = self.f.readinto(b)
            if n:
                return n
            self.f.close()
            self.i += 1
            if self.i >= len(self.paths):
                return 0
            self.f = open(self.paths[self.i], 'rb')


# ----------------------------------------------------------------------------------------------- trace parsing

def trace_points(o):
    """Columns of one trace (all points) with callsign forward-filled."""
    tr = o.get('trace') or []
    n = len(tr)
    ts0 = float(o.get('timestamp') or 0.0)
    t = np.empty(n)
    lat = np.empty(n)
    lon = np.empty(n)
    alt = np.full(n, np.nan)
    gnd = np.zeros(n, bool)
    gs = np.full(n, np.nan)
    trk = np.full(n, np.nan)
    flg = np.zeros(n, np.int16)
    vr = np.full(n, np.nan)
    src = [''] * n
    cs = [''] * n
    cur = ''
    for i, p in enumerate(tr):
        t[i] = ts0 + p[0]
        lat[i] = p[1]
        lon[i] = p[2]
        a = p[3]
        if a == 'ground':
            gnd[i] = True
        elif a is not None:
            alt[i] = a
        if p[4] is not None:
            gs[i] = p[4]
        if p[5] is not None:
            trk[i] = p[5]
        if p[6]:
            flg[i] = p[6]
        if p[7] is not None:
            vr[i] = p[7]
        q = p[8]
        if q:
            f = q.get('flight')
            if f is not None:
                cur = f.strip()
        cs[i] = cur
        if len(p) > 9 and p[9]:
            src[i] = p[9]
    return dict(t=t, lat=lat, lon=lon, alt=alt, gnd=gnd, gs=gs, trk=trk, flg=flg, vr=vr,
                src=np.array(src, dtype=object), cs=np.array(cs, dtype=object))


def select_near(c):
    """Indices of points worth keeping and their airport index."""
    k, d = nearest_airport(c['lat'], c['lon'])
    el = AP_EL[k]
    keep_g = c['gnd'] & (d <= R_GROUND)
    keep_a = (~c['gnd']) & (d <= R_AIR) & (np.isnan(c['alt']) | (c['alt'] <= el + ALT_AIR))
    idx = np.nonzero(keep_g | keep_a)[0]
    return idx, k[idx], d[idx]


# ----------------------------------------------------------------------------------------------- events

def local_xy(lat, lon, la, lo):
    return (lon - lo) * 111200.0 * math.cos(math.radians(la)), (lat - la) * 111200.0


def still(gs_k, dd):
    """A point counts as standing still: speed <= STAT_KT, or below MOVE_KT (or unknown) within STOP_R of the
    stop anchor. A point at MOVE_KT or faster never stands still, however close it is."""
    if math.isnan(gs_k):
        return dd <= STOP_R
    if gs_k <= STAT_KT:
        return dd <= 2.5 * STOP_R
    return gs_k < MOVE_KT and dd <= STOP_R


def stops_of(t, x, y, gs):
    """Runs of consecutive standing-still points (see still()) around the run's first point.
    Returns list of (first index, last index); a run of one moving point is returned as (k, k) too."""
    n = len(t)
    out = []
    k = 0
    while k < n:
        s = k
        if still(gs[s], 0.0):
            while k + 1 < n and still(gs[k + 1], math.hypot(x[k + 1] - x[s], y[k + 1] - y[s])):
                k += 1
        out.append((s, k))
        k += 1
    return out


def is_stop(t, gs, s, e):
    return still(gs[s], 0.0) and (e > s or (not math.isnan(gs[s]) and gs[s] <= STAT_KT))


def first_motion_after(t, x, y, gs, s, e):
    """First point after stop (s..e), i.e. the first point that moves away from it."""
    j = e + 1
    return j if j < len(t) else None


def parked_gap(gs, x, y, p, a):
    """A long gap between points p and a keeps the segment only if the aircraft stood still on both sides at
    the same place (parked with the transponder off); any other long gap is an unobserved movement."""
    def slow(v):
        return math.isnan(v) or v < MOVE_KT
    return slow(gs[p]) and slow(gs[a]) and math.hypot(x[a] - x[p], y[a] - y[p]) <= 2.5 * STOP_R


def is_real_air(alt, gs, el):
    return (not math.isnan(gs) and gs >= 50) or (not math.isnan(alt) and alt >= el + 300)


def events_for(c, ap):
    """Takeoffs at airport ap from the kept points c (sorted by time) of one aircraft."""
    la, lo, el = AIRPORTS[ap]
    t, gnd, alt, gs, vr, flg = c['t'], c['gnd'], c['alt'], c['gs'], c['vr'], c['flg']
    x, y = local_xy(c['lat'], c['lon'], la, lo)
    n = len(t)
    out = []
    last_to = -1
    for i in range(1, n):
        if not (gnd[i - 1] and not gnd[i]):
            continue
        # confirm climb-out: no ground point and a climb within TO_CONFIRM s
        ok = False
        j = i
        while j < n and t[j] - t[i] <= TO_CONFIRM and not gnd[j]:
            if ((not math.isnan(alt[j])) and (not math.isnan(alt[i])) and alt[j] - alt[i] >= 300) or \
               ((not math.isnan(gs[j])) and gs[j] >= 100) or ((not math.isnan(vr[j])) and vr[j] >= 600):
                ok = True
                break
            j += 1
        if not ok:
            continue
        # walk back over the ground segment that ends at i-1
        a = i - 1
        reason = 'trace_start'
        while a - 1 > last_to:
            p = a - 1
            if not gnd[p]:
                if is_real_air(alt[p], gs[p], el):
                    reason = 'landing'
                    break
            gap = t[a] - t[p]
            if gap >= GAP_CUT and not parked_gap(gs, x, y, p, a):
                reason = 'gap_cut'
                break
            a = p
        seg = np.arange(a, i)
        seg = seg[gnd[seg]]
        if seg.size == 0:
            continue
        last_to = i
        tt, xx, yy, gg = t[seg], x[seg], y[seg], gs[seg]
        st = stops_of(tt, xx, yy, gg)
        # stand stop
        stand = None
        if reason == 'landing':
            for s_, e_ in st:
                if is_stop(tt, gg, s_, e_) and tt[e_] - tt[s_] >= TURN_STOP:
                    stand = (s_, e_)
                    break
        else:
            s_, e_ = st[0]
            if is_stop(tt, gg, s_, e_):
                stand = (s_, e_)
        fm = None
        if stand is not None:
            fm = first_motion_after(tt, xx, yy, gg, stand[0], stand[1])
        t_fm = float(tt[fm]) if fm is not None else np.nan
        t_stand_end = float(tt[fm - 1]) if fm is not None else np.nan
        # stops between first motion and takeoff
        mids = []
        if fm is not None:
            for s_, e_ in st:
                if s_ >= fm and is_stop(tt, gg, s_, e_):
                    d_ = tt[e_] - tt[s_]
                    if d_ >= MID_STOP:
                        mids.append(d_)
        # alternative stand: the last stop of at least LONG_STOP s anywhere in the segment (tow, then park)
        longs = [(s_, e_) for s_, e_ in st if is_stop(tt, gg, s_, e_) and tt[e_] - tt[s_] >= LONG_STOP]
        t_fm_last = fm_last_gap = last_long_dur = np.nan
        if longs:
            s_, e_ = longs[-1]
            j_ = first_motion_after(tt, xx, yy, gg, s_, e_)
            if j_ is not None:
                t_fm_last = float(tt[j_])
                fm_last_gap = float(tt[j_] - tt[j_ - 1])
                last_long_dur = float(tt[e_] - tt[s_])
        after = slice(fm if fm is not None else 0, len(tt))
        gaps = np.diff(tt[after]) if len(tt[after]) > 1 else np.array([0.0])
        # callsign: most common non-empty in [t_to - 600, t_to + 600]
        w = (t >= t[i] - 600) & (t <= t[i] + 600)
        vals = [v for v in c['cs'][w] if v]
        if vals:
            u, k_ = np.unique(np.array(vals), return_counts=True)
            cs = str(u[np.argmax(k_)])
        else:
            cs = c['cs'][i] or ''
        # heading on the take-off roll: track of the last ground points with gs > 40 kt
        roll = [k for k in range(max(0, i - 15), i) if gnd[k] and not math.isnan(gs[k]) and gs[k] > 40 and not math.isnan(c['trk'][k])]
        roll_trk = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(c['trk'][roll]))), np.mean(np.cos(np.radians(c['trk'][roll])))))) % 360 if roll else np.nan
        src_seg = c['src'][seg]
        out.append(dict(
            apt=ap, cs=cs, t_to=float(t[i]), t_last_ground=float(t[i - 1]), to_gap=float(t[i] - t[i - 1]),
            roll_trk=roll_trk, seg_reason=reason,
            t_first=float(tt[0]), first_lat=float(c['lat'][seg[0]]), first_lon=float(c['lon'][seg[0]]),
            first_gs=float(gg[0]) if not math.isnan(gg[0]) else np.nan,
            first_dist_m=float(math.hypot(xx[0], yy[0])), first_src=str(src_seg[0]),
            first_stationary=stand is not None and reason != 'landing',
            stand_seen=stand is not None,
            stand_dur=float(tt[stand[1]] - tt[stand[0]]) if stand is not None else np.nan,
            stand_lat=float(c['lat'][seg[stand[0]]]) if stand is not None else np.nan,
            stand_lon=float(c['lon'][seg[stand[0]]]) if stand is not None else np.nan,
            t_stand_end=t_stand_end, t_fm=t_fm, fm_gap=float(t_fm - t_stand_end) if fm is not None else np.nan,
            n_mid_stops=len(mids), mid_stop_s=float(sum(mids)), mid_stop_max=float(max(mids)) if mids else 0.0,
            n_long_stops=len(longs), t_fm_last=t_fm_last, fm_last_gap=fm_last_gap, last_long_dur=last_long_dur,
            n_pts=int(seg.size), max_gap_after_fm=float(gaps.max()),
            share_adsb=float(np.mean(src_seg == 'adsb_icao')), share_mlat=float(np.mean(src_seg == 'mlat')),
        ))
    return out


# ----------------------------------------------------------------------------------------------- main

def cpu_s():
    return time.process_time()


def max_rss_mb():
    if resource is not None:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 2**20 if sys.platform == 'darwin' else r / 1024, 1)
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        p = PMC()
        p.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL('kernel32')
        psapi = ctypes.WinDLL('psapi')
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb)
        return round(p.PeakWorkingSetSize / 2**20, 1)
    except Exception:
        return None


def events_from_points(path):
    """Re-derive events from a day's points.parquet (no raw archive needed)."""
    import pandas as pd
    p = pd.read_parquet(path)
    p['src'] = p['src'].astype(str)
    events = []
    for (icao, apn), g in p.groupby(['icao', 'apt'], sort=False):
        g = g.sort_values('t', kind='stable')
        c = dict(t=g.t.to_numpy(float), lat=g.lat.to_numpy(float), lon=g.lon.to_numpy(float),
                 alt=g.alt.to_numpy(float), gnd=g.gnd.to_numpy(bool), gs=g.gs.to_numpy(float),
                 trk=g.trk.to_numpy(float), flg=g.flg.to_numpy(np.int16), vr=g.vr.to_numpy(float),
                 src=g.src.to_numpy(object), cs=g.cs.fillna('').to_numpy(object))
        for e in events_for(c, apn):
            e.update(icao=icao, reg=g.reg.iat[0], typ=g.typ.iat[0])
            events.append(e)
    return events


def download(parts, rawdir, log):
    os.makedirs(rawdir, exist_ok=True)
    rec = []
    for url, size in parts:
        fn = os.path.join(rawdir, url.rsplit('/', 1)[1])
        t0 = time.time()
        for attempt in range(5):
            have = os.path.getsize(fn) if os.path.exists(fn) else 0
            if have == size:
                break
            subprocess.run(['curl', '-sSL', '--retry', '5', '--retry-delay', '5', '-C', '-', '-o', fn, url], check=False)
        have = os.path.getsize(fn) if os.path.exists(fn) else 0
        dt = time.time() - t0
        rec.append(dict(part=os.path.basename(fn), bytes=have, want=size, seconds=round(dt, 1),
                        mb_s=round(have / 1e6 / dt, 1) if dt > 0 else None))
        print('download', rec[-1], file=log, flush=True)
        if have != size:
            raise SystemExit(f'download failed {fn} {have} != {size}')
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--manifest', default='manifest.tsv')
    ap.add_argument('--out', default='out')
    ap.add_argument('--raw', default='raw')
    ap.add_argument('--keep-raw', action='store_true')
    ap.add_argument('--stream', action='store_true')
    ap.add_argument('--no-download', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--from-points', action='store_true', help='only re-derive events.parquet from points.parquet')
    ap.add_argument('--nice', action='store_true', help='run at below-normal priority (Windows) or nice 10')
    a = ap.parse_args()
    day = a.day
    if a.nice:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL('kernel32', use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    if a.from_points:
        outdir = os.path.join(a.out, day)
        c0, w0 = cpu_s(), time.time()
        events = events_from_points(os.path.join(outdir, 'points.parquet'))
        pq.write_table(pa.Table.from_pylist(events), os.path.join(outdir, 'events.parquet'), compression='zstd')
        mf = os.path.join(outdir, 'meta.json')
        meta = json.load(open(mf)) if os.path.exists(mf) else {}
        meta.update(n_events=len(events), events_by_apt={k: sum(1 for e in events if e['apt'] == k) for k in AIRPORTS},
                    events_cpu_s=round(cpu_s() - c0, 1), events_wall_s=round(time.time() - w0, 1))
        json.dump(meta, open(mf, 'w'), indent=1)
        print(day, 'events', len(events), meta['events_by_apt'], 'cpu', meta['events_cpu_s'])
        return
    parts = []
    with open(a.manifest, encoding='utf-8') as f:
        for line in f:
            d, tag, repo, names, sizes = line.rstrip('\n').split('\t')[:5]
            if d == day:
                for nm, sz in zip(names.split(','), sizes.split(',')):
                    parts.append((f'https://github.com/adsblol/{repo}/releases/download/{tag}/{nm}', int(sz)))
    if not parts:
        raise SystemExit(f'{day} not in manifest')
    outdir = os.path.join(a.out, day)
    rawdir = os.path.join(a.raw, day)
    os.makedirs(outdir, exist_ok=True)
    log = sys.stdout
    meta = dict(day=day, parts=[p[0] for p in parts], bytes=sum(p[1] for p in parts))
    t_wall0 = time.time()
    if a.stream:
        proc = subprocess.Popen(['curl', '-sSL', '--retry', '5'] + [p[0] for p in parts], stdout=subprocess.PIPE,
                                bufsize=16 << 20)
        stream = io.BufferedReader(proc.stdout, buffer_size=16 << 20)
        meta['mode'] = 'stream'
    else:
        if not a.no_download:
            meta['download'] = download(parts, rawdir, log)
            meta['download_s'] = round(sum(r['seconds'] for r in meta['download']), 1)
            meta['download_mb_s'] = round(meta['bytes'] / 1e6 / meta['download_s'], 1)
        files = [os.path.join(rawdir, p[0].rsplit('/', 1)[1]) for p in parts]
        stream = io.BufferedReader(MultiFile(files), buffer_size=16 << 20)
        meta['mode'] = 'disk'
    rx = build_prefilter()
    cpu0, t0 = cpu_s(), time.time()
    st = dict(members=0, traces=0, gz_bytes=0, raw_bytes=0, has_ground=0, prefilter_hits=0, kept_traces=0)
    other = {}
    texts = {}
    keys = ('icao', 'reg', 'typ', 'apt', 't', 'lat', 'lon', 'alt', 'gnd', 'gs', 'trk', 'flg', 'vr', 'src', 'cs')
    cols = {k: [] for k in keys}
    schema = pa.schema([('icao', pa.string()), ('reg', pa.string()), ('typ', pa.string()), ('apt', pa.string()),
                        ('t', pa.float64()), ('lat', pa.float64()), ('lon', pa.float64()), ('alt', pa.float32()),
                        ('gnd', pa.bool_()), ('gs', pa.float32()), ('trk', pa.float32()), ('flg', pa.int16()),
                        ('vr', pa.float32()), ('src', pa.string()), ('cs', pa.string())])
    pts_path = os.path.join(outdir, 'points.parquet')
    writer = pq.ParquetWriter(pts_path + '.part', schema, compression='zstd')
    n_points = 0
    pending = 0

    def flush():
        nonlocal n_points, pending
        if not cols['t']:
            return
        def cat(k):
            return np.concatenate(cols[k])
        tab = pa.table({
            'icao': pa.array(cols['icao'], pa.string()), 'reg': pa.array(cols['reg'], pa.string()),
            'typ': pa.array(cols['typ'], pa.string()), 'apt': pa.array(cols['apt'], pa.string()),
            't': pa.array(cat('t'), pa.float64()), 'lat': pa.array(cat('lat'), pa.float64()),
            'lon': pa.array(cat('lon'), pa.float64()), 'alt': pa.array(cat('alt'), pa.float32()),
            'gnd': pa.array(cat('gnd'), pa.bool_()), 'gs': pa.array(cat('gs'), pa.float32()),
            'trk': pa.array(cat('trk'), pa.float32()), 'flg': pa.array(cat('flg'), pa.int16()),
            'vr': pa.array(cat('vr'), pa.float32()),
            'src': pa.array(list(cat('src')), pa.string()), 'cs': pa.array(list(cat('cs')), pa.string()),
        }, schema=schema)
        writer.write_table(tab)
        n_points += tab.num_rows
        for k in keys:
            cols[k].clear()
        pending = 0

    kept_ac = {k: 0 for k in AIRPORTS}
    tf = tarfile.open(fileobj=stream, mode='r|')
    for m in tf:
        st['members'] += 1
        if not m.isfile():
            continue
        name = m.name
        if 'trace_full_' not in name:
            top = name.lstrip('./').split('/')[0]
            other[top] = other.get(top, 0) + 1
            if top.upper().startswith(('README', 'LICENSE')) and m.size < 200_000:
                texts[top] = tf.extractfile(m).read().decode('utf-8', 'replace')
            continue
        st['traces'] += 1
        gz = tf.extractfile(m).read()
        st['gz_bytes'] += len(gz)
        try:
            d = gunzip(gz) if gz[:2] == b'\x1f\x8b' else gz
        except Exception as ex:  # a corrupt member in the published archive (2026-01-29 staging): skip that aircraft
            st['bad_members'] = st.get('bad_members', 0) + 1
            st.setdefault('bad_member_names', []).append(f'{name}: {type(ex).__name__} {ex}'[:200])
            print('skip corrupt member', name, type(ex).__name__, ex, file=log, flush=True)
            continue
        st['raw_bytes'] += len(d)
        if b'"ground"' not in d:
            continue
        st['has_ground'] += 1
        if not rx.search(d):
            continue
        st['prefilter_hits'] += 1
        o = orjson.loads(d)
        c = trace_points(o)
        if len(c['t']) == 0:
            continue
        idx, kap, _ = select_near(c)
        if idx.size == 0:
            continue
        st['kept_traces'] += 1
        icao = o.get('icao', '')
        reg = o.get('r') or ''
        typ = o.get('t') or ''
        sub = {k: v[idx] for k, v in c.items()}
        aps = [AP_LIST[k] for k in kap]
        for k_ in np.unique(kap):
            if sub['gnd'][kap == k_].any():
                kept_ac[AP_LIST[k_]] += 1
        nn = idx.size
        cols['icao'].extend([icao] * nn)
        cols['reg'].extend([reg] * nn)
        cols['typ'].extend([typ] * nn)
        cols['apt'].extend(aps)
        for k in ('t', 'lat', 'lon', 'alt', 'gnd', 'gs', 'trk', 'flg', 'vr', 'src', 'cs'):
            cols[k].append(sub[k])
        pending += 1
        if pending >= 300:
            flush()
        if a.limit and st['traces'] >= a.limit:
            break
        if st['traces'] % 20000 == 0:
            print(f"{st['traces']} traces {st['prefilter_hits']} hits {st['gz_bytes'] / 1e9:.2f} GB gz "
                  f"{time.time() - t0:.0f}s", file=log, flush=True)
    if a.stream:
        proc.wait()
    flush()
    writer.close()
    os.replace(pts_path + '.part', pts_path)
    st['parse_wall_s'] = round(time.time() - t0, 1)
    st['parse_cpu_s'] = round(cpu_s() - cpu0, 1)
    st['kept_aircraft_ground'] = kept_ac
    st['other_members'] = other
    meta.update(st)
    # events from the written points (same code path as --from-points)
    c1 = cpu_s()
    events = events_from_points(pts_path)
    if events:
        pq.write_table(pa.Table.from_pylist(events), os.path.join(outdir, 'events.parquet'), compression='zstd')
    meta['events_cpu_s'] = round(cpu_s() - c1, 1)
    meta['wall_s'] = round(time.time() - t_wall0, 1)
    meta['max_rss_mb'] = max_rss_mb()
    meta['n_points'] = n_points
    meta['n_events'] = len(events)
    meta['events_by_apt'] = {k: sum(1 for e in events if e['apt'] == k) for k in AIRPORTS}
    for k, v in texts.items():
        with open(os.path.join(outdir, k), 'w', encoding='utf-8') as f:
            f.write(v)
    with open(os.path.join(outdir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ('parts',)}, indent=None), file=log, flush=True)
    if not a.stream and not a.keep_raw and os.path.isdir(rawdir):
        shutil.rmtree(rawdir)


if __name__ == '__main__':
    main()
