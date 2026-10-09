# SPDX-License-Identifier: GPL-3.0-only
"""R18-M3 selection (registered in the private note before any Muon number): among the fwd runs of tabm_r18c.py --opt muon with the three
learning rates, pick the one with the lowest early-stopping-day RMSE (days inside the training months January to June; the July ruler rows
are not used). Ties go to the middle rate. Prints the chosen tag and writes research/r18/logs/m3_choice.txt.
"""
import json
from pathlib import Path

R = Path(__file__).parent
cands = [("r18tmmu004", 0.004), ("r18tmmu012", 0.012), ("r18tmmu036", 0.036)]
es = {}
for tag, lr in cands:
    f = R / "runs" / f"{tag}_fwd_s0.json"
    es[tag] = json.loads(f.read_text(encoding="utf-8"))["es_rmse"] if f.exists() else float("inf")
best = min(es.values())
pick = [t for t, _ in cands if es[t] == best]
tag = "r18tmmu012" if len(pick) > 1 and "r18tmmu012" in pick else pick[0]
lr = dict(cands)[tag]
(R / "logs" / "m3_choice.txt").write_text(f"{tag} {lr}\n{json.dumps(es)}\n", encoding="utf-8")
print(tag, lr)
