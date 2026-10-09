"""F3 fit cache around unchanged leg scripts (coordinator approval 2026-10-08 23:2x KST). Run a leg script once while its training data is
final but its test rows are not, keep every fitted model; when the final test rows exist, run the same unchanged script again and every
fit whose key matches exactly is loaded instead of refit. The script's code is not touched: this file runs it (runpy, as `python
script.py` would: same sys.argv, sys.path[0] = the script's directory, __main__) and only intercepts the fit calls.

Key of a fit = blake2b of everything that fixes it:
  LightGBM  lightgbm.train(params, train_set, num_boost_round): the training Dataset's identity (every lightgbm.Dataset argument: the data
            array or DataFrame by content, label, weight, group, init_score, feature names, categorical spec, Dataset params, reference;
            for a subset, the parent's identity, the indices and the subset params), the label, weight, init_score and group as LightGBM
            holds them at train time (for a subset not built yet, the parent's), the Dataset params, feature names and categorical spec at
            train time, every train parameter (seed and num_threads included), num_boost_round, the lightgbm version and this file's sha16.
            A fit given valid_sets, init_model, feval, callbacks, keep_training_booster or extra positional arguments is never cached.
  CatBoost  CatBoostRegressor.fit(Pool): every Pool argument (data by content, label, cat_features, weight, baseline, ...), every model
            parameter (get_params: iterations, seed, thread_count ...), the catboost version and this file's sha16. A fit given anything
            but the Pool (eval_set, y, ...) is never cached.
Data of an unsupported type makes the fit uncacheable (it runs normally). A model is loaded only on an exact key match; otherwise mode
load fits normally (and stores the model), mode strict stops the run (tests). Each stored model has a sidecar <key>.json with its key
components and per-column data hashes; on a miss the components that differ from the stored fit of the same script, arguments and call
number are logged (so a column whose training values moved is named). Events: <dir>/fitwrap_log.jsonl. Any edit of this file changes every
key (its sha16 is part of the key).
    python -X utf8 -u research/f3/fitwrap.py <save|load|strict> <cache dir> <script.py | -m module> [script args ...]
"""
from __future__ import annotations

import hashlib
import inspect
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODE, DIR = sys.argv[1], Path(sys.argv[2]).resolve()
assert MODE in ("save", "load", "strict"), MODE
DIR.mkdir(parents=True, exist_ok=True)
LOG = DIR / "fitwrap_log.jsonl"
SALT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
STATS = {"saved": 0, "loaded": 0, "miss_fit": 0, "uncached": 0}
RUN: dict = {"target": None, "args": None, "seq": {"lgb": 0, "cb": 0}}


def log(ev: dict) -> None:
    ev["t"] = time.strftime("%m-%d %H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, default=str) + "\n")


def _jd(o):
    """JSON fallback inside parameters and lists: arrays by full content (never a truncated repr), numpy scalars as numbers."""
    if isinstance(o, (np.ndarray, pd.Series, pd.Index)):
        return "nd:" + hb(np.asarray(o))
    if isinstance(o, pd.DataFrame):
        return "df:" + str(frame_parts(o)[0])
    if isinstance(o, np.generic):
        return o.item()
    return f"{type(o).__name__}:{o!s}"


def hb(*parts) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if p is None:
            h.update(b"<None>")
        elif isinstance(p, np.ndarray):
            a = np.ascontiguousarray(p)
            h.update(repr((a.dtype.str, a.shape)).encode())
            if a.dtype == object:
                h.update(json.dumps(a.tolist(), default=_jd).encode())
            else:
                h.update(memoryview(a.reshape(-1).view(np.uint8)))
        elif isinstance(p, (bytes, bytearray)):
            h.update(bytes(p))
        else:
            try:
                h.update(json.dumps(p, sort_keys=True, default=_jd).encode())
            except TypeError:
                h.update(repr(p).encode())
        h.update(b"|")
    return h.hexdigest()


def frame_parts(x):
    """(hash, per-column hashes) of a 2-d ndarray or a DataFrame by content; (None, None) for any other type."""
    if isinstance(x, np.ndarray):
        if x.ndim != 2:
            return hb("nd", x), [hb(x)]
        cols = [hb(np.ascontiguousarray(x[:, j])) for j in range(x.shape[1])]
        return hb("nd2", x.dtype.str, list(x.shape), cols), cols
    if isinstance(x, pd.DataFrame):
        cols = []
        for c in x.columns:
            s = x[c]
            if isinstance(s.dtype, pd.CategoricalDtype):
                cols.append(hb("cat", str(c), [str(v) for v in s.cat.categories], bool(s.cat.ordered), s.cat.codes.to_numpy()))
            elif isinstance(s.dtype, np.dtype) and s.dtype.kind in "biufcmM":
                cols.append(hb("col", str(c), s.dtype.str, s.to_numpy()))
            else:
                cols.append(hb("obj", str(c), str(s.dtype), pd.util.hash_pandas_object(s, index=False).to_numpy()))
        return hb("df", [str(c) for c in x.columns], list(x.shape), cols), cols
    return None, None


def val_hash(v):
    if v is None:
        return None
    if isinstance(v, (np.ndarray, pd.Series, pd.Index)):
        return hb(np.asarray(v))
    if isinstance(v, (list, tuple)) and v and all(isinstance(e, (np.ndarray, pd.Series)) for e in v):
        return hb(*[np.asarray(e) for e in v])
    return hb(v)


def sidecar_diff(lib: str, comps: dict, cols: list | None) -> dict:
    """Components that differ from the stored fit of the same script, arguments and call number (if one is stored)."""
    for p in DIR.glob(f"{lib}_*.json"):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if s.get("target") == RUN["target"] and s.get("args") == RUN["args"] and s.get("seq") == RUN["seq"][lib]:
            d = sorted(k for k in set(comps) | set(s["components"]) if comps.get(k) != s["components"].get(k))
            out = {"stored_key": s["key"], "components_differing": d}
            if cols is not None and s.get("data_cols") is not None:
                sc = s["data_cols"]
                names = s.get("data_col_names") or list(range(len(sc)))
                out["data_columns_differing"] = [names[j] for j in range(min(len(sc), len(cols))) if sc[j] != cols[j]] + (
                    ["<column count differs>"] if len(sc) != len(cols) else [])
            return out
    return {"stored_key": None}


def lookup_or_fit(lib: str, comps: dict, cols, col_names, fit, load, store, ntrees):
    key = f"{lib}_" + hb(sorted(comps.items()))
    p = DIR / (f"{key}.txt" if lib == "lgb" else f"{key}.cbm")
    seq = RUN["seq"][lib]
    if MODE in ("load", "strict") and p.exists():
        STATS["loaded"] += 1
        log({"lib": lib, "event": "loaded", "key": key, "seq": seq})
        return load(p)
    if MODE in ("load", "strict"):
        diff = sidecar_diff(lib, comps, cols)
        ev = "MISS_STOP" if MODE == "strict" else "MISS_fit"
        log({"lib": lib, "event": ev, "key": key, "seq": seq, **diff})
        print(f"fitwrap {ev} {lib} call {seq}: {json.dumps(diff, default=str)[:1500]}", file=sys.stderr, flush=True)
        if MODE == "strict":
            raise SystemExit(f"fitwrap strict: no stored model for {lib} fit {key}")
        STATS["miss_fit"] += 1
    r = fit()
    store(p)
    (DIR / f"{key}.json").write_text(json.dumps({"key": key, "lib": lib, "target": RUN["target"], "args": RUN["args"], "seq": seq,
                                                 "salt": SALT, "components": comps, "data_cols": cols, "data_col_names": col_names,
                                                 "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, default=str), encoding="utf-8")
    STATS["saved"] += 1
    log({"lib": lib, "event": "saved", "key": key, "seq": seq, "trees": ntrees()})
    return r


# ---------------- LightGBM ----------------
import lightgbm as lgb  # noqa: E402

_ds_init, _ds_subset, _train = lgb.Dataset.__init__, lgb.Dataset.subset, lgb.train
_ds_sig = inspect.signature(_ds_init)


def ds_init(self, *a, **kw):
    self._f3h, self._f3cols, self._f3names = None, None, None
    try:
        b = _ds_sig.bind(self, *a, **kw)
        b.apply_defaults()
        g = dict(b.arguments)
        g.pop("self", None)
        dh, cols = frame_parts(g.pop("data", None))
        ref = g.pop("reference", None)
        if dh is not None and (ref is None or getattr(ref, "_f3h", None) is not None):
            comps = {"data": dh, "reference": getattr(ref, "_f3h", None) if ref is not None else None}
            comps.update({k: val_hash(v) for k, v in g.items()})
            self._f3h, self._f3cols = hb("ds", sorted(comps.items())), cols
            d = b.arguments.get("data")
            self._f3names = [str(c) for c in d.columns] if isinstance(d, pd.DataFrame) else None
    except Exception as e:  # identity unknown: the fit runs uncached
        self._f3h = None
        log({"lib": "lgb", "event": "identity_error", "error": repr(e)})
    _ds_init(self, *a, **kw)


def ds_subset(self, used_indices, params=None):
    r = _ds_subset(self, used_indices, params)
    p = getattr(self, "_f3h", None)
    r._f3h = None if p is None else hb("subset", p, np.asarray(used_indices), val_hash(params))
    r._f3cols, r._f3names = getattr(self, "_f3cols", None), getattr(self, "_f3names", None)
    return r


def ds_fields(ds) -> dict:
    """label, weight, init_score, group as LightGBM holds them now (built: the C++ fields; not built: what was given)."""
    if getattr(ds, "_handle", None) is not None:
        return {"label": val_hash(ds.get_label()), "weight": val_hash(ds.get_weight()), "init_score": val_hash(ds.get_init_score()),
                "group": val_hash(ds.get_group()), "built": True}
    return {"label": val_hash(ds.label), "weight": val_hash(ds.weight), "init_score": val_hash(ds.init_score), "group": val_hash(ds.group),
            "built": False}


def train(params, train_set, num_boost_round=100, *a, **kw):
    RUN["seq"]["lgb"] += 1
    risky = bool(a) or any(kw.get(k) for k in ("valid_sets", "init_model", "feval", "callbacks", "keep_training_booster"))
    ident = getattr(train_set, "_f3h", None)
    if risky or ident is None:
        STATS["uncached"] += 1
        log({"lib": "lgb", "event": "uncached", "seq": RUN["seq"]["lgb"], "reason": "valid_sets/callbacks/args" if risky else "no identity"})
        return _train(params, train_set, num_boost_round, *a, **kw)
    comps = {"dataset": ident, "train_params": val_hash(params), "rounds": int(num_boost_round), "lib_version": lgb.__version__, "salt": SALT,
             "ds_params": val_hash(getattr(train_set, "params", None)), "ds_feature_name": val_hash(getattr(train_set, "feature_name", None)),
             "ds_categorical": val_hash(getattr(train_set, "categorical_feature", None)), "kwargs": val_hash(sorted(kw))}
    comps.update({f"now_{k}": v for k, v in ds_fields(train_set).items()})
    ref = getattr(train_set, "reference", None)
    if getattr(train_set, "_handle", None) is None and getattr(train_set, "used_indices", None) is not None and ref is not None:
        comps.update({f"parent_now_{k}": v for k, v in ds_fields(ref).items()})
    holder = {}

    def fit():
        holder["m"] = _train(params, train_set, num_boost_round, *a, **kw)
        return holder["m"]

    def store(p):
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(holder["m"].model_to_string(), encoding="utf-8")
        tmp.replace(p)

    return lookup_or_fit("lgb", comps, train_set._f3cols, train_set._f3names, fit,
                         lambda p: lgb.Booster(model_str=p.read_text(encoding="utf-8")), store, lambda: holder["m"].num_trees())


lgb.Dataset.__init__, lgb.Dataset.subset, lgb.train = ds_init, ds_subset, train

# ---------------- CatBoost ----------------
try:
    import catboost  # noqa: E402
    from catboost import CatBoostRegressor, Pool  # noqa: E402
except ImportError:
    catboost = None

if catboost is not None:
    _pool_init, _cb_fit = Pool.__init__, CatBoostRegressor.fit
    _pool_sig = inspect.signature(_pool_init)

    def pool_init(self, *a, **kw):
        h = cols = names = None
        try:
            b = _pool_sig.bind(self, *a, **kw)
            b.apply_defaults()
            g = dict(b.arguments)
            g.pop("self", None)
            for k in ("log_cout", "log_cerr"):
                g.pop(k, None)
            d = g.pop("data", None)
            dh, cols = frame_parts(d)
            if dh is not None:
                comps = {"data": dh}
                comps.update({k: val_hash(v) for k, v in g.items()})
                h = hb("pool", sorted(comps.items()))
                names = [str(c) for c in d.columns] if isinstance(d, pd.DataFrame) else None
        except Exception as e:
            h = None
            log({"lib": "cb", "event": "identity_error", "error": repr(e)})
        _pool_init(self, *a, **kw)
        self._f3h, self._f3cols, self._f3names = h, cols, names

    def cb_fit(self, X, *a, **kw):
        RUN["seq"]["cb"] += 1
        ident = getattr(X, "_f3h", None) if isinstance(X, Pool) else None
        if a or ident is None or set(kw) - {"verbose", "logging_level"}:
            STATS["uncached"] += 1
            log({"lib": "cb", "event": "uncached", "seq": RUN["seq"]["cb"]})
            return _cb_fit(self, X, *a, **kw)
        comps = {"pool": ident, "model_params": val_hash(self.get_params()), "lib_version": catboost.__version__, "salt": SALT}

        def load(p):
            self.load_model(str(p))
            return self

        def store(p):
            tmp = p.with_name(p.name + ".tmp")
            self.save_model(str(tmp))
            tmp.replace(p)

        return lookup_or_fit("cb", comps, X._f3cols, X._f3names, lambda: _cb_fit(self, X, *a, **kw), load, store, lambda: self.tree_count_)

    Pool.__init__, CatBoostRegressor.fit = pool_init, cb_fit


def main():
    target, rest = sys.argv[3], sys.argv[4:]
    RUN["target"], RUN["args"] = (rest[0] if target == "-m" else target), (rest[1:] if target == "-m" else rest)
    log({"event": "start", "mode": MODE, "salt": SALT, "target": RUN["target"], "args": RUN["args"], "cwd": str(Path.cwd()),
         "lightgbm": lgb.__version__, "catboost": getattr(catboost, "__version__", None)})
    rc = "ok"
    try:
        if target == "-m":
            sys.path[0] = str(Path.cwd())
            sys.argv = [rest[0]] + rest[1:]
            runpy.run_module(rest[0], run_name="__main__", alter_sys=True)
        else:
            sys.path[0] = str(Path(target).resolve().parent)
            sys.argv = [target] + rest
            runpy.run_path(str(Path(target).resolve()), run_name="__main__")
    except SystemExit as e:
        rc = f"SystemExit({e.code})"
        raise
    except BaseException as e:
        rc = repr(e)[:300]
        raise
    finally:
        log({"event": "end", "stats": STATS, "rc": rc})
        print(f"fitwrap {MODE} {DIR}: {STATS} salt {SALT}", flush=True)


if __name__ == "__main__":
    main()
