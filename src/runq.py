# SPDX-License-Identifier: GPL-3.0-only
"""명령 줄 목록(파일 한 줄에 '로그파일<TAB>명령')을 동시에 최대 N개(기본 2)로 돌린다. 끝날 때마다 한 줄 남긴다.

사용: python src/runq.py jobs.txt [N]
"""
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    jobs = [ln.split("\t", 1) for ln in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    running = []
    env = dict(__import__("os").environ)
    env["PYTHONIOENCODING"] = "utf-8"
    while jobs or running:
        while jobs and len(running) < n:
            log, cmd = jobs.pop(0)
            f = open(ROOT / log, "w", encoding="utf-8")
            argv = shlex.split(cmd)
            if argv[0].replace("\\", "/").endswith("python.exe"):
                argv[0] = sys.executable  # Windows CreateProcess 는 상대 경로 실행 파일을 못 찾는다
            p = subprocess.Popen(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
            running.append((p, f, log, time.time()))
            print(f"start {log}", flush=True)
        time.sleep(5)
        for item in list(running):
            p, f, log, t0 = item
            if p.poll() is not None:
                f.close()
                running.remove(item)
                print(f"done {log} rc={p.returncode} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
