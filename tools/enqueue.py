# SPDX-License-Identifier: GPL-3.0-only
# manifest 하나를 큐 후보로 넣는다(측정 묶음 등록 전 단계). 시험 예측을 바이트 그대로 queue/<이름>.parquet 로 복사하고, json 에 manifest 와
# 홀드아웃을 다시 잰 로컬 값을 적는다. manifest 검사(해시, config 와 commands)를 통과해야 넣는다.
#     .venv/Scripts/python.exe -X utf8 tools/enqueue.py preds/manifest_<spec>.json <큐 이름> "<method 설명>" [--readiness-warn]
# 예측에 NaN 이 있거나 제출 틀 행이 빠졌으면 늘 거부한다(src/shot.py pred_problems).
# --readiness-warn(10-05 단일 발 체제, 준비 점검은 경고): 준비 점검이 걸려도 거부하지 않고 큐 json 에 "readiness_warn": true 와 걸린 칸
#     (readiness_failed: check, value, band)을 적어 넣는다. 그래도 시험에서 통째로 빈 피처 칸이 있으면 거부하고(src/shot.py empty_on_test),
#     준비 점검이 끝나지 않았으면(메모리 상한 등) 결과가 없으므로 거부한다. 플래그가 없으면 예전처럼 걸리면 거부한다.
import json
import os
import shutil
import subprocess
import time
import sys
from pathlib import Path

# numpy 를 들이는 순간 OpenBLAS 가 스레드마다 버퍼를 잡는다(이 PC 에서 프로세스당 약 350 MiB). 행렬 곱을 쓰지 않으니 한 스레드로 막고, 준비 점검 자식도 물려받는다
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import shot as S  # noqa: E402
from src.batches import QUEUE, hold, manifest_of, sha  # noqa: E402
from src.ruler import rmse_of  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main(mpath: str, name: str, method: str, warn: bool = False) -> int:
    m, why = manifest_of({"manifest": mpath})
    if why:
        print("거부:", why)
        return 1
    dst = QUEUE / f"{name}.parquet"
    if dst.exists() or dst.with_suffix(".json").exists():
        print("거부: 같은 이름의 큐 항목이 있다", dst.name)
        return 1
    shutil.copy2(ROOT / m["test"]["path"], dst)
    assert sha(f"queue/{name}.parquet") == m["test"]["sha"]
    bad = S.pred_problems(dst)
    if bad:
        dst.unlink()
        print("거부: 예측 파일이 제출할 수 없는 모양이다(큐에 넣지 않았다)", bad)
        return 1
    rm = rmse_of(m["holdout"]["path"], hold())
    meta = {"kind": "CANDIDATE", "method": method, "local_rmse": round(rm, 3), "manifest": mpath,
            "note": f"spec {m['spec']}, feats {m['feats']}, seeds {m['seeds']}, 홀드아웃 {m['holdout']['path']}(다시 잰 값 {rm:.3f})"}
    dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    # 준비 점검을 넣을 때 바로 돌린다: 등록 때 처음 걸리면 그 사이 나온 점수를 본 뒤 묶음을 다시 짜게 된다(10-05 묶음 004, 181 과 191 과 231)
    out = f"notes/composition_{name}.json"
    t0 = time.time()
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "src.composition", "--only", name, "--out", out], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    outp = ROOT / out
    fresh = outp.exists() and outp.stat().st_mtime >= t0 - 1  # 점검이 중간에 죽으면(메모리 상한 등) 옛 출력이나 빈 자리를 읽지 않는다
    if r.returncode != 0 or not fresh:
        comp = json.loads(outp.read_text(encoding="utf-8")) if fresh else {}
        fails = S.readiness_failed(comp)
        hard = S.empty_problems(S.empty_on_test(m["feats"], m["feats_sha"])) if (warn and fresh and fails) else []
        if warn and fresh and fails and not hard:
            meta.update(readiness=out, readiness_warn=True, readiness_failed=fails)
            dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            print("큐", dst.name, "local", round(rm, 3), "spec", m["spec"], "준비 점검 미통과, 경고로 기록(--readiness-warn)", out,
                  [(c["check"], c["value"], c["band"]) for c in fails][:4])
            return 0
        dst.unlink()
        dst.with_suffix(".json").unlink()
        what = ("미통과" + (", 시험에서 통째로 빈 칸 " + "; ".join(hard) if hard else "")) if fails else \
            f"이 끝나지 않았다(rc {r.returncode}, 새 출력 {fresh}): {(r.stderr or '')[-300:]}"
        print(f"거부: 준비 점검 {what}(큐에 넣지 않았다)", out, [(c["check"], c["value"], c["band"]) for c in fails][:4])
        return 1
    meta["readiness"] = out
    dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("큐", dst.name, "local", round(rm, 3), "spec", m["spec"], "준비 점검 통과", out)
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--readiness-warn"]
    if len(args) != 3:
        print("usage: enqueue.py <manifest> <name> <method> [--readiness-warn]")
        sys.exit(2)
    sys.exit(main(args[0], args[1], args[2], warn="--readiness-warn" in sys.argv[1:]))
