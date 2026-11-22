#!/usr/bin/env python3
"""
복합(동시) 공격 대응 평가 (sim/.venv): 복합 공격 3종 + 단일 공격 9종(부작용 확인)
조건: 방어 없음 / 단일 진단(가장 의심되는 센서 하나만 보정) / 순차 진단(보정 후에도 이상이면 다음 센서 추가)
공격당 15회 (공격 시작 20/30/40 s × 시드 5), 공격 후 40 s.
출력: out/combo_results.json, out/v2_runs/combo_*.csv
"""
import json
import os

import numpy as np
import pandas as pd

from defense_v2 import ATTACKS, COMBOS, OUT_DIR, Detector, make_attack, run

STARTS, SEEDS, LEN = [20.0, 30.0, 40.0], [0, 1, 2, 3, 4], 40.0
RUNS = os.path.join(OUT_DIR, "v2_runs")


def main():
    d = Detector("v2")
    names = list(COMBOS) + list(ATTACKS)
    rows = []
    for ai, n in enumerate(names):
        tgt = set(make_attack(n, 0).target.split("+"))
        for si, st in enumerate(STARTS):
            for sd in SEEDS:
                seed = 8000 + 100 * ai + 10 * si + sd
                for cond in ("none", "z", "seq"):
                    df, sim = run(n, st, seed, detector=None if cond == "none" else d, diag=cond if cond != "none" else "z", length=LEN)
                    sel = set(sim.diag_group.split("+")) if sim.diag_group else set()
                    rows.append({"attack": n, "combo": n in COMBOS, "start": st, "seed": seed, "cond": cond,
                                 "xte": float(df[df.t >= st].xte.mean()),
                                 "det_t": sim.detect_t, "pre": sim.detect_t is not None and sim.detect_t < st,
                                 "diag": sim.diag_group, "exact": sel == tgt, "covers": tgt <= sel,
                                 "steps": [(float(a), b) for a, b in sim.diag_times]})
                    if n in COMBOS and st == 30.0 and sd == 0:
                        df.to_csv(os.path.join(RUNS, f"combo_{n}_{cond}.csv"), index=False)
        print("done", n, flush=True)
    R = pd.DataFrame(rows)
    summ = []
    for n in names:
        g = R[R.attack == n]
        r = {"attack": n, "combo": n in COMBOS, "target": make_attack(n, 0).target}
        for c in ("none", "z", "seq"):
            h = g[g.cond == c]
            r[f"xte_{c}"] = round(h.xte.mean(), 3)
            if c != "none":
                r[f"exact_{c}"] = int(h.exact.sum()); r[f"covers_{c}"] = int(h.covers.sum()); r["n"] = len(h)
        summ.append(r); print(r)
    with open(os.path.join(OUT_DIR, "combo_results.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summ, "runs": rows}, f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
