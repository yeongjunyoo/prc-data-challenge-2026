# SPDX-License-Identifier: GPL-3.0-only
"""OpenSky MinIO(S3) 입출력: 대회 데이터 받기, 제출 올리기, 채점 결과 읽기.

The access key is read from the environment variables PRC_S3_LOGIN_ID and PRC_S3_PASSWORD and is never printed.
    python -m src.s3io ls [bucket]
    python -m src.s3io pull            # prc-2026-datasets 전부를 data/ 로(크기가 같으면 건너뜀)
    python -m src.s3io push <file>     # 팀 버킷에 올린다(이름은 spirited-tiger_vN.parquet, 결선 허가로는 spirited-tiger_final.parquet 하나)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import boto3
from botocore.config import Config

ENDPOINT = "https://s3.opensky-network.org"
DATA_BUCKET = "prc-2026-datasets"
TEAM = "spirited-tiger"
TEAM_BUCKET = f"prc-2026-{TEAM}"
FINAL_NAME = f"{TEAM}_final.parquet"  # ranking.html 결선 절: “<team-name>_final.parquet”(10-08)
ROOT = Path(__file__).resolve().parent.parent


def _cred(field: str) -> str:
    """Access key of your own MinIO account from the environment (PRC_S3_LOGIN_ID, PRC_S3_PASSWORD). Nothing is stored in this repository."""
    import os
    name = {"login_id": "PRC_S3_LOGIN_ID", "password": "PRC_S3_PASSWORD"}[field]
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"set the environment variable {name} (your own challenge access key)")
    return v


def client():
    return boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=_cred("login_id"),
                        aws_secret_access_key=_cred("password"), region_name="us-east-1",
                        config=Config(signature_version="s3v4", retries={"max_attempts": 5}))


def ls(bucket: str = DATA_BUCKET) -> list[dict]:
    s3 = client()
    out, token = [], None
    while True:
        kw = {"Bucket": bucket, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        out += [{"key": o["Key"], "size": o["Size"], "modified": o["LastModified"].isoformat()} for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def head(bucket: str, key: str) -> dict:
    """객체 하나의 ETag(따옴표 없이), 크기, 수정 시각. 읽기만 한다(HeadObject). 결선 자료 고정값과 올린 결선 파일 확인에 쓴다."""
    h = client().head_object(Bucket=bucket, Key=key)
    return {"etag": str(h["ETag"]).strip('"'), "size": int(h["ContentLength"]), "modified": h["LastModified"].isoformat()}


def pull(dest: Path = ROOT / "data") -> None:
    s3 = client()
    dest.mkdir(parents=True, exist_ok=True)
    for o in ls(DATA_BUCKET):
        p = dest / o["key"]
        if p.exists() and p.stat().st_size == o["size"]:
            print("skip", o["key"])
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(DATA_BUCKET, o["key"], str(p))
        print("got", o["key"], o["size"])


DAILY_QUOTA = 5  # ranking.html: “There is currently a limit of 5 submissions per day”


def quota_used_today() -> int:
    """오늘(UTC 날짜) 쓴 제출 수. 받는 쪽 기록(팀 버킷의 _result.json 시각과 올린 제출 파일 시각) 중 큰 쪽이다.
    채점이 밀려 결과가 없는 제출(10-09 v40)도 한 칸을 쓴 것이라 src/shot.py bucket_state 와 같은 식으로 센다."""
    import datetime as dt
    today = dt.datetime.now(dt.timezone.utc).date()
    s3 = client()
    r = s3.list_objects_v2(Bucket=TEAM_BUCKET)
    objs = [o for o in r.get("Contents", []) if o["LastModified"].astimezone(dt.timezone.utc).date() == today]
    results = sum(1 for o in objs if o["Key"].endswith("_result.json"))
    uploads = sum(1 for o in objs if o["Key"].startswith(f"{TEAM}_v") and o["Key"].endswith(".parquet"))
    return max(results, uploads)


def last_result_at():
    """가장 최근 채점 결과(_result.json)의 시각(UTC). 없으면 None."""
    r = client().list_objects_v2(Bucket=TEAM_BUCKET)
    times = [o["LastModified"] for o in r.get("Contents", []) if o["Key"].endswith("_result.json")]
    return max(times) if times else None


def result(name: str) -> dict:
    import json
    body = client().get_object(Bucket=TEAM_BUCKET, Key=f"{name}_result.json")["Body"].read()
    return json.loads(body)


PERMIT = ROOT / "notes" / "push_permit.json"


def push(path: Path) -> None:
    """제출 올리기. src/fire_queue.py 가 게이트, 측정 묶음, 쿼터를 통과시킨 뒤 쓴 일회용 허가(파일 이름과 sha256)가 있어야 올린다.
    CLI 나 다른 스크립트는 허가가 없어 올리지 못한다(private notes §0 G2, private notes §5). 허가는 확인하는 즉시 지운다.
    10-05 부터 GATE.json firing 이 single 이면 CANDIDATE 는 fire-one 의 단일 발 허가(mode single)로만 올라가고 src/shot.py upload_problem 이
    다시 본다. 그동안 묶음 허가는 거부한다(묶음 경로는 남겨 두되 쓰지 않는다).
    결선(10-08 공지): spirited-tiger_final.parquet 하나만, fire-final 이 쓴 결선 허가(kind FINAL)로만 올라가고 src/shot.py final_upload_problem 이
    다시 본다. 버킷에 결선 파일이 있으면 허가에 replace 가 있을 때만 덮어쓴다. 한 번에 올리게(PUT 하나) 해서 ETag 가 파일 MD5 와 같다."""
    import hashlib
    import json
    path = Path(path)
    final = path.name == FINAL_NAME
    if "final" in path.name.lower() and not final:
        raise SystemExit(f"결선 파일 이름은 {FINAL_NAME} 하나뿐이다: {path.name}")
    if not final and not (path.name.startswith(f"{TEAM}_v") and path.suffix == ".parquet"):
        raise SystemExit(f"제출 이름은 {TEAM}_vN.parquet 또는 {FINAL_NAME} 이어야 한다: {path.name}")
    import datetime as dt
    permit = json.loads(PERMIT.read_text(encoding="utf-8")) if PERMIT.exists() else {}
    PERMIT.unlink(missing_ok=True)
    if permit.get("file") != path.name or permit.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise SystemExit("발사 허가가 없다: 제출은 src/fire_queue.py 만 올린다(GATE.json, 측정 묶음, 장부)")
    age = dt.datetime.now(dt.timezone.utc).timestamp() - float(permit.get("issued_utc_ts", 0))
    if not 0 <= age <= 600:
        raise SystemExit(f"발사 허가가 만료됐다({age:.0f}초, 10분 안이어야 한다)")
    gate = json.loads((ROOT / "GATE.json").read_text(encoding="utf-8"))
    kind = permit.get("kind")
    if final != (kind == "FINAL"):
        raise SystemExit("결선 파일은 결선 허가(kind FINAL)로만, 결선 허가는 결선 파일로만 올린다")
    ok = {"CANDIDATE": bool(gate.get("candidate_allowed")), "ANCHOR": not gate.get("anchor_fired"),
          "DIAG": gate.get("diag_used", 0) < gate.get("diag_limit", 0),
          "FINAL": bool(gate.get("candidate_allowed")) and gate.get("firing") == "single"}.get(kind, False)
    if not ok:
        raise SystemExit(f"지금 GATE.json 이 {kind} 발을 허락하지 않는다")
    if kind == "FINAL":
        from src.shot import final_upload_problem  # 결선: 자료 고정값, 결선 틀, intent, 후보, 값, 막는 것 셋, 쿼터, 장부 밖 제출, 버킷의 결선 파일
        why = final_upload_problem(path, permit)
        if why:
            raise SystemExit(f"올리기 직전 확인 실패(결선): {why}")
        if FINAL_NAME in {o["key"] for o in ls(TEAM_BUCKET)} and not permit.get("replace"):
            raise SystemExit(f"팀 버킷에 {FINAL_NAME} 이 이미 있다(덮어쓰려면 fire-final --replace-final)")
        from boto3.s3.transfer import TransferConfig
        client().upload_file(str(path), TEAM_BUCKET, FINAL_NAME, Config=TransferConfig(multipart_threshold=256 * 1024 * 1024))
        print("pushed", FINAL_NAME, "(final)")
        return
    if kind == "CANDIDATE":
        if permit.get("mode") == "single":
            from src.shot import upload_problem as single_problem  # 단일 발: intent, 후보, 값, 막는 것 셋, 엿보기 막이, 쿼터, 장부 밖 제출을 다시 본다
            why = single_problem(path, permit)
        elif gate.get("firing") == "single":
            why = "묶음 경로는 은퇴했다(GATE.json firing single, 10-05). 발사는 python -m src.fire_queue fire-one 으로만 한다"
        else:
            from src.batches import upload_problem  # 허가가 손으로 만든 것이어도 지금 차례인 등록 후보의 예측만 올라간다(재검토 5판)
            why = upload_problem(path, permit)
        if why:
            raise SystemExit(f"올리기 직전 확인 실패: {why}")
    existing = {o["key"] for o in ls(TEAM_BUCKET)}
    if path.name in existing:
        raise SystemExit(f"이미 올린 이름이다(덮어쓰지 않는다): {path.name}")
    client().upload_file(str(path), TEAM_BUCKET, path.name)
    print("pushed", path.name)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1]
    if cmd == "ls":
        for o in ls(sys.argv[2] if len(sys.argv) > 2 else DATA_BUCKET):
            print(o["modified"][:19], f'{o["size"]:>12,}', o["key"])
    elif cmd == "pull":
        pull()
    elif cmd == "push":
        push(Path(sys.argv[2]))
    elif cmd == "head":
        print(head(sys.argv[2], sys.argv[3]))
    elif cmd == "quota":
        used = quota_used_today()
        print(f"used {used} / {DAILY_QUOTA} (UTC today), left {DAILY_QUOTA - used}")
    elif cmd == "result":
        print(result(sys.argv[2]))
