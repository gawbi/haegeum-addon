#!/usr/bin/env python3
"""v2 정상 데이터: 학습 200 s × 6회(시드 42~47), 검증 120 s × 2회(시드 123, 124). 시동 구간이 여러 번 들어가도록 나눠 주행."""
import os
import pandas as pd
from defense_v2 import FEATURES_V2, OUT_DIR, SimV2

for name, seeds, dur in (("normal_train_v2", range(42, 48), 200.0), ("normal_test_v2", (123, 124), 120.0)):
    dfs = []
    for s in seeds:
        d = SimV2(seed=s).run(dur); d["run"] = s; dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=False)
    print(name, len(df), "rows")
print(df[FEATURES_V2].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
