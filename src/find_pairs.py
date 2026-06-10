from pathlib import Path
import argparse
import csv
import re
import sys
import time
import shutil
import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "third_party/accelerated_features")
from modules.xfeat import XFeat

# Selected snow/query frames
SELECTED_IDS = ["00000", "00036", "00073", "00133", "00691"]
# Paths
ROOT = Path("data/cmu_seasons/images_raw/images")
SNOW_LIST = Path("docs/cmu_snow_21dec2010/snow_date_images.txt")
OUT = Path("docs/snow_pair_search")
CARDS = OUT / "cards"
DEMO = Path("data/demo_pairs_snow")


def sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# Parse camera id from filename
def get_cam(path):
    m = re.search(r"_c([01])_", path.name)
    if not m:
        raise ValueError(f"Could not parse camera from {path}")
    return f"c{m.group(1)}"


def load_img_tensor(path, device, max_side):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ten = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
    return img, ten.to(device)


def to_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[0]
    return x


@torch.inference_mode()
def match_pair(xfeat, db_path, snow_path, device, max_side, top_k, ransac_thresh):
    db_img, db_t = load_img_tensor(db_path, device, max_side)
    snow_img, snow_t = load_img_tensor(snow_path, device, max_side)
    try:
        pts0, pts1 = xfeat.match_xfeat_star(db_t, snow_t, top_k=top_k)
    except Exception:
        pts0, pts1 = xfeat.match_xfeat_star(db_img, snow_img, top_k=top_k)
    sync_cuda(device)
    pts0 = to_np(pts0)
    pts1 = to_np(pts1)
    if len(pts0) < 4:
        return 0, 0, 0.0, 0.0
    H, mask = cv2.findHomography(
        pts0,
        pts1,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=1000,
        confidence=0.99,
    )
    if H is None or mask is None:
        return len(pts0), 0, 0.0, 0.0
    mask = mask.reshape(-1).astype(bool)
    matches = int(len(pts0))
    inliers = int(mask.sum())
    ratio = float(inliers / matches) if matches else 0.0
    score = inliers * ratio
    return matches, inliers, ratio, score


# Save candidate preview card
def draw_card(db_path, snow_path, matches, inliers, ratio, score, out_path):
    db = cv2.imread(str(db_path))
    snow = cv2.imread(str(snow_path))

    def resize(img, w=520):
        h0, w0 = img.shape[:2]
        scale = w / w0
        return cv2.resize(img, (w, int(h0 * scale)), interpolation=cv2.INTER_AREA)

    db = resize(db)
    snow = resize(snow)
    h = max(db.shape[0], snow.shape[0])
    w = db.shape[1] + snow.shape[1] + 40
    canvas = np.ones((h + 120, w, 3), dtype=np.uint8) * 255
    canvas[: db.shape[0], : db.shape[1]] = db
    canvas[: snow.shape[0], db.shape[1] + 40 : db.shape[1] + 40 + snow.shape[1]] = snow
    cv2.putText(
        canvas,
        "SPRING/REFERENCE DATABASE",
        (15, h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        canvas,
        "WINTER/SNOW QUERY",
        (db.shape[1] + 55, h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        canvas,
        f"matches={matches} | inliers={inliers} | ratio={ratio:.3f} | score={score:.1f}",
        (15, h + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        canvas,
        f"DB: {db_path.parent.parent.name}/{db_path.parent.name}/{db_path.name[:45]}",
        (15, h + 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        canvas,
        f"SNOW: {snow_path.parent.parent.name}/{snow_path.parent.name}/{snow_path.name[:45]}",
        (db.shape[1] + 55, h + 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
    )
    cv2.imwrite(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_db", type=int, default=900)
    ap.add_argument("--max_side", type=int, default=640)
    ap.add_argument("--top_k", type=int, default=1200)
    ap.add_argument("--ransac_thresh", type=float, default=4.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CARDS.mkdir(parents=True, exist_ok=True)
    (DEMO / "summer").mkdir(parents=True, exist_ok=True)
    (DEMO / "winter").mkdir(parents=True, exist_ok=True)
    id_to_path = {}
    for line in SNOW_LIST.read_text().splitlines():
        idx, path = line.split(",", 1)
        id_to_path[idx] = Path(path)
    snow_paths = [id_to_path[i] for i in SELECTED_IDS]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    xfeat = XFeat(top_k=args.top_k).eval()
    all_rows = []
    final_rows = []
    for pair_idx, snow_path in enumerate(snow_paths, start=1):
        slice_name = snow_path.parent.parent.name
        cam = get_cam(snow_path)
        db_folder = ROOT / slice_name / "database"
        db_paths = sorted(db_folder.glob(f"*_{cam}_*rect.jpg"))
        if args.max_db and len(db_paths) > args.max_db:
            idxs = np.linspace(0, len(db_paths) - 1, args.max_db).astype(int)
            db_paths = [db_paths[i] for i in idxs]
        print(f"\nSearching match for {snow_path.name}")
        print(f"Slice={slice_name}, camera={cam}, database candidates={len(db_paths)}")
        rows = []
        start = time.perf_counter()
        for i, db_path in enumerate(db_paths, start=1):
            matches, inliers, ratio, score = match_pair(
                xfeat,
                db_path,
                snow_path,
                device,
                args.max_side,
                args.top_k,
                args.ransac_thresh,
            )
            row = {
                "snow_candidate_id": SELECTED_IDS[pair_idx - 1],
                "database_path": str(db_path),
                "snow_path": str(snow_path),
                "matches": matches,
                "inliers": inliers,
                "ratio": ratio,
                "score": score,
                "slice": slice_name,
                "camera": cam,
            }
            rows.append(row)
            all_rows.append(row)
            if i % 100 == 0:
                print(f"  checked {i}/{len(db_paths)}")
        rows = sorted(rows, key=lambda r: r["score"], reverse=True)
        best = rows[0]
        print(
            f"Best: matches={best['matches']} | inliers={best['inliers']} | "
            f"ratio={best['ratio']:.3f} | score={best['score']:.1f}"
        )
        for rank, r in enumerate(rows[:5], start=1):
            card_path = (
                CARDS
                / f"snow_{SELECTED_IDS[pair_idx-1]}_rank_{rank:02d}_ratio_{r['ratio']:.2f}.jpg"
            )
            draw_card(
                Path(r["database_path"]),
                Path(r["snow_path"]),
                r["matches"],
                r["inliers"],
                r["ratio"],
                r["score"],
                card_path,
            )
        pair_id = f"pair_{pair_idx:03d}"
        summer_out = DEMO / "summer" / f"{pair_id}.jpg"
        winter_out = DEMO / "winter" / f"{pair_id}.jpg"
        shutil.copy(best["database_path"], summer_out)
        shutil.copy(best["snow_path"], winter_out)
        final_rows.append(
            {
                "pair_id": pair_id,
                "summer_path": str(summer_out),
                "winter_path": str(winter_out),
                "source_snow_candidate": SELECTED_IDS[pair_idx - 1],
                "source_matches": best["matches"],
                "source_inliers": best["inliers"],
                "source_ratio": best["ratio"],
                "source_score": best["score"],
            }
        )
        print(f"Time for this snow image: {time.perf_counter() - start:.1f}s")
    pd.DataFrame(all_rows).to_csv(OUT / "all_snow_match_results.csv", index=False)
    pd.DataFrame(final_rows).to_csv(DEMO / "pairs.csv", index=False)
    print("\nSaved candidate cards:", CARDS)
    print("Saved final snow pairs:", DEMO / "pairs.csv")
    print(pd.DataFrame(final_rows))


if __name__ == "__main__":
    main()
