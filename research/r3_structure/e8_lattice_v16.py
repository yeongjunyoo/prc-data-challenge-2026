# SPDX-License-Identifier: GPL-3.0-only
"""(1) 분 격자 제약: EDDM, EGLL, LFPG, LIRF, LTFM 은 y mod 60 이 (MVT − 가까운 분) 의 ±6초 안, LEBL, LEMD 는 (MVT mod 60) 의 ±6초 안.
321 예측을 가장 가까운 격자 이빨 중심으로 옮기면 자 MSE 가 어떻게 되나(전부, M1 행만).
(2) G 이음 반대 창 규칙의 값을 v16 바탕(서버 266.31, 창 판)에서도 잰다."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")
S = pd.read_parquet(OUT / "e1_rows.parquet")
N = 344419
A = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "t_mvt"]).to_pandas().set_index("MVT_ID_mvt")["t_mvt"]
t = A.reindex(S.index).to_numpy()
y = S["y"].to_numpy(); p = S["p_c321"].to_numpy(); ap = S["ap"].astype(str).to_numpy()
jm = t - 60 * np.round(t / 60)          # 5곳: MVT 의 흔들기
c5 = np.isin(ap, ["EDDM", "EGLL", "LFPG", "LIRF", "LTFM"]); c2 = np.isin(ap, ["LEBL", "LEMD"])
center = np.where(c5, jm, np.where(c2, np.mod(t, 60), np.nan))   # y ≡ center (mod 60) ± 6
ok = np.isfinite(center)
r = (y - center)
dev = r - 60 * np.round(r / 60)
print(f"격자 칸 행 {ok.sum():,}: |y − 이빨 중심| ≤ 6.5 비율 {float((np.abs(dev[ok]) <= 6.5).mean()):.4f}")
snap = p.copy()
rp = p - center
snap[ok] = (center + 60 * np.round(rp / 60))[ok]
se0 = ((p - y) ** 2); se1 = ((snap - y) ** 2)
print(f"모든 격자 행을 가까운 이빨로: 자 ΔMSE {(se1.sum() - se0.sum()) / N:+.1f}")
m1 = ok & S["fm_ok"].to_numpy()
print(f"M1(움직임 시작 보임) 격자 행 {int(m1.sum()):,}만: 자 ΔMSE {(se1[m1].sum() - se0[m1].sum()) / N:+.1f}, 그 행 RMSE {np.sqrt(se0[m1].mean()):.1f} → {np.sqrt(se1[m1].mean()):.1f}")
# (2) v16 바탕에서 G 이음 규칙
L = pd.read_parquet(OUT / "s5_link_train_unmatched.parquet").set_index("MVT_ID_mvt")
b16 = pd.read_parquet(ROOT / "cache" / "val_m6r25sgcbxrw_s0.parquet").set_index("MVT_ID_mvt")
src = pq.read_table(ROOT / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "src"]).to_pandas().set_index("MVT_ID_mvt")["src"]
L["src"] = src.reindex(L.index).to_numpy()
tr = L[~L["src"].isin([1, 7])]
c = float(np.median((tr["y"] - tr["d_aobt_a"])[np.abs(tr["sch_m_lobt"]) <= 3606]))
for nm, b in (("v16", b16), ("321", pd.read_parquet(ROOT / "cache" / "val_m6r25sgcbxradsbgeow_s0.parquet").set_index("MVT_ID_mvt"))):
    J = L[L["src"].isin([1, 7])].join(b[["pred", "g"]], how="inner")
    okr = (np.abs(J["sch_m_lobt"]) <= 3606) & (J["g"] == 1) & np.isfinite(J["d_aobt_a"])
    new = J["pred"].where(~okr, J["d_aobt_a"] + c)
    d = (((new - J["y"]) ** 2).sum() - ((J["pred"] - J["y"]) ** 2).sum()) / N
    print(f"G 이음 규칙 on {nm}: 대상 {int(okr.sum())}, 자 ΔMSE {d:+,.1f}")
