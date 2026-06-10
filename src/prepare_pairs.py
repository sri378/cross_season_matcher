# File utilities
from pathlib import Path
import shutil
import pandas as pd

choices = {
    "00000": 1,
    "00073": 1,
    "00133": 1,
    "00036": 3,
    "00691": 1,
}
results = pd.read_csv("docs/snow_pair_search/all_snow_match_results.csv")
# Pair preparation
out_root = Path("data/demo_pairs_snow_manual")
summer = out_root / "summer"
winter = out_root / "winter"
summer.mkdir(parents=True, exist_ok=True)
winter.mkdir(parents=True, exist_ok=True)
rows = []
for out_idx, (snow_id, rank_no) in enumerate(choices.items(), start=1):
    group = results[results["snow_candidate_id"].astype(str).str.zfill(5) == snow_id]
    group = group.sort_values("score", ascending=False).reset_index(drop=True)
    row = group.iloc[rank_no - 1]
    pair_id = f"pair_{out_idx:03d}"
    summer_out = summer / f"{pair_id}.jpg"
    winter_out = winter / f"{pair_id}.jpg"
    shutil.copy(row["database_path"], summer_out)
    shutil.copy(row["snow_path"], winter_out)
    rows.append(
        {
            "pair_id": pair_id,
            "summer_path": str(summer_out),
            "winter_path": str(winter_out),
            "snow_candidate_id": snow_id,
            "chosen_rank": rank_no,
            "source_matches": int(row["matches"]),
            "source_inliers": int(row["inliers"]),
            "source_ratio": float(row["ratio"]),
            "source_score": float(row["score"]),
        }
    )
df = pd.DataFrame(rows)
df.to_csv(out_root / "pairs.csv", index=False)
print(df)
print(f"\nSaved final snow pairs to: {out_root / 'pairs.csv'}")
