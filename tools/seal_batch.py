# SPDX-License-Identifier: GPL-3.0-only
# 측정 묶음의 봉인(notes/seals/<번호>.json, src/batches.py write_seal)을 쓰고 외부 시각 둘을 받는다: RFC 3161 시각 증명(기관 서명, openssl 로 바로 검증, 재검토 6판)과
# OpenTimestamps(비트코인 확정은 몇 시간 뒤, 재검토 5판).
# 공개 달력 서버로 가는 것은 봉인 파일 다이제스트에 무작위 16바이트를 붙여 다시 해시한 32바이트뿐이다. 증명은 notes/seals/<번호>.json.ots 이고,
# 비트코인 확정은 몇 시간 뒤 --upgrade 로 받는다. `ots` 명령은 이 호스트에서 python-bitcoinlib 의 OpenSSL 로드(ctypes)에서 죽어서, 같은 일을
# opentimestamps 라이브러리로 직접 한다(otsclient stamp 와 같은 절차: 파일 sha256, OpAppend(nonce), OpSHA256, 달력 제출, 병합, 직렬화).
# 발사와 등록(src/batches.py ledger_state)은 봉인의 해시와 지금 파일, 봉인 파일과 .ots 다이제스트를 맞춘다. 외부 시각은 닫힘 감사가 본다.
#     .venv/Scripts/python.exe -X utf8 tools/seal_batch.py <번호> [--upgrade]
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from opentimestamps.calendar import RemoteCalendar
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.op import OpAppend, OpSHA256
from opentimestamps.core.serialize import StreamDeserializationContext, StreamSerializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.batches import BATCHES, OPENSSL, SEALS, TSA, seal_problems, tsa_check, write_seal  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
CALENDARS = ("https://a.pool.opentimestamps.org", "https://b.pool.opentimestamps.org",
             "https://a.pool.eternitywall.com", "https://ots.btc.catallaxy.com")  # otsclient 기본 달력


def stamp(sp: Path, ots: Path) -> list[str]:
    with open(sp, "rb") as f:
        dtf = DetachedTimestampFile.from_fd(OpSHA256(), f)
    tip = dtf.timestamp.ops.add(OpAppend(os.urandom(16))).ops.add(OpSHA256())
    ok = []
    for url in CALENDARS:
        try:
            tip.merge(RemoteCalendar(url).submit(tip.msg, timeout=30))
            ok.append(url)
        except Exception as e:  # noqa: BLE001  달력 하나가 죽어도 나머지로 간다
            print("달력 실패", url, repr(e)[:120])
    if not ok:
        raise SystemExit("어느 달력도 받지 않았다. 증명을 쓰지 않았다")
    with open(ots, "xb") as f:
        dtf.serialize(StreamSerializationContext(f))
    return ok


def load(ots: Path) -> DetachedTimestampFile:
    with open(ots, "rb") as f:
        return DetachedTimestampFile.deserialize(StreamDeserializationContext(f))


def nodes(t):
    yield t
    for sub in t.ops.values():
        yield from nodes(sub)


def upgrade(ots: Path) -> int:
    dtf, n = load(ots), 0
    for msg, att in list(dtf.timestamp.all_attestations()):
        if isinstance(att, PendingAttestation):
            try:
                sub = RemoteCalendar(att.uri).get_timestamp(msg, timeout=30)
            except Exception as e:  # noqa: BLE001  아직 확정 전이면 달력이 404 를 준다
                print("아직", att.uri, repr(e)[:80])
                continue
            for stamp_ in nodes(dtf.timestamp):
                if stamp_.msg == msg:
                    stamp_.merge(sub)
                    n += 1
                    break
    if n:
        tmp = ots.with_suffix(".ots.tmp")
        with open(tmp, "wb") as f:
            dtf.serialize(StreamSerializationContext(f))
        tmp.replace(ots)
    return n


def tsa_stamp(sp: Path, name: str) -> str:
    """봉인 파일의 sha256 으로 RFC 3161 질의를 만들어 시각 기관에 보내고 응답(.tsr)을 한 번만 쓴다. 보내는 것은 다이제스트뿐이다."""
    tok = sp.parent / f"{sp.name}.{name}.tsr"
    if tok.exists():
        return "이미 있다"
    q = sp.parent / f"{sp.name}.{name}.tsq"
    subprocess.run([OPENSSL, "ts", "-query", "-data", str(sp), "-sha256", "-cert", "-out", str(q)], check=True, capture_output=True)
    req = urllib.request.Request(TSA[name]["url"], data=q.read_bytes(), headers={"Content-Type": "application/timestamp-query"})
    body = urllib.request.urlopen(req, timeout=30).read()
    with open(tok, "xb") as f:
        f.write(body)
    return f"{len(body)}바이트"


n = int(sys.argv[1])
entry = json.loads((BATCHES / f"{n:03d}.json").read_text(encoding="utf-8"))
sp = SEALS / f"{n:03d}.json"
if not sp.exists():
    write_seal(entry)
    print("봉인을 썼다", sp.relative_to(ROOT).as_posix())
ots = sp.parent / (sp.name + ".ots")
if not ots.exists():
    print("달력이 받았다:", ", ".join(stamp(sp, ots)))
elif "--upgrade" in sys.argv:
    print("확정을 받아 합친 증명:", upgrade(ots))
if ots.exists():
    d = load(ots)
    print("증명:", ots.relative_to(ROOT).as_posix(), "파일 sha256", d.file_digest.hex())
    for _, att in d.timestamp.all_attestations():
        print("  ", "비트코인 블록 " + str(att.height) if isinstance(att, BitcoinBlockHeaderAttestation) else f"대기 {getattr(att, 'uri', att)}")
for name in TSA:
    print(f"{name} 시각 증명:", tsa_stamp(sp, name), "검증:", tsa_check(sp, name))
print("봉인 검사:", seal_problems(sp) or "이상 없음")
