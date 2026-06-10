from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

# feature-matching dependency
sys.path.insert(0, "third_party/accelerated_features")
from modules.xfeat import XFeat


# CUDA timing helper
def sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def resize_bgr(path, max_side):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img


def bgr_to_tensor(img_bgr, device):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ten = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
    return ten.to(device)


def to_numpy_points(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def get_star_matches(xfeat, img0_bgr, img1_bgr, ten0, ten1, top_k):
    try:
        pts0, pts1 = xfeat.match_xfeat_star(ten0, ten1, top_k=top_k)
    except Exception:
        pts0, pts1 = xfeat.match_xfeat_star(img0_bgr, img1_bgr, top_k=top_k)
    pts0 = to_numpy_points(pts0)
    pts1 = to_numpy_points(pts1)
    if pts0.ndim == 3:
        pts0 = pts0[0]
    if pts1.ndim == 3:
        pts1 = pts1[0]
    return pts0, pts1


def compute_homography(pts0, pts1, ransac_thresh, max_iters, confidence):
    if len(pts0) < 4 or len(pts1) < 4:
        return np.eye(3), np.zeros((len(pts0),), dtype=bool), "not_enough_matches"
    # Robust geometric verification
    H, mask = cv2.findHomography(
        pts0.astype(np.float32),
        pts1.astype(np.float32),
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=max_iters,
        confidence=confidence,
    )
    if H is None or mask is None:
        return np.eye(3), np.zeros((len(pts0),), dtype=bool), "homography_failed"
    return H, mask.reshape(-1).astype(bool), "ok"


def draw_matches(img0, img1, pts0, pts1, inlier_mask, pair_id, max_lines=350):
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas_h = max(h0, h1) + 70
    canvas_w = w0 + w1
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    canvas[:h0, :w0] = img0
    canvas[:h1, w0 : w0 + w1] = img1
    if len(pts0) > 0:
        if inlier_mask.any():
            indices = np.where(inlier_mask)[0]
        else:
            indices = np.arange(len(pts0))
        if len(indices) > max_lines:
            indices = indices[np.linspace(0, len(indices) - 1, max_lines).astype(int)]
        for idx in indices:
            p0 = tuple(np.round(pts0[idx]).astype(int))
            p1 = tuple(np.round(pts1[idx]).astype(int))
            p1_shifted = (p1[0] + w0, p1[1])
            color = (0, 255, 0) if inlier_mask[idx] else (0, 0, 255)
            cv2.line(canvas, p0, p1_shifted, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p1_shifted, 2, color, -1, cv2.LINE_AA)
    ratio = float(inlier_mask.mean()) if len(inlier_mask) else 0.0
    text = f"{pair_id} | XFeat* semi-dense | matches={len(pts0)} | inliers={int(inlier_mask.sum())} | ratio={ratio:.3f}"
    cv2.putText(
        canvas,
        text,
        (20, canvas_h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def build_clean_warp(img0, img1, H, pair_id):
    h1, w1 = img1.shape[:2]
    warped = cv2.warpPerspective(img0, H, (w1, h1))
    overlay = cv2.addWeighted(img1, 0.60, warped, 0.40, 0)
    diff = cv2.absdiff(img1, warped)
    label_h = 60
    strip_h = h1 + label_h
    strip_w = w1 * 4
    strip = np.ones((strip_h, strip_w, 3), dtype=np.uint8) * 255
    strip[:h1, 0:w1] = img1
    strip[:h1, w1 : 2 * w1] = warped
    strip[:h1, 2 * w1 : 3 * w1] = overlay
    strip[:h1, 3 * w1 : 4 * w1] = diff
    labels = [
        f"{pair_id} | target",
        "warped source",
        "overlay proof",
        "absolute difference",
    ]
    for i, label in enumerate(labels):
        cv2.putText(
            strip,
            label,
            (i * w1 + 18, strip_h - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 255),
            2,
        )
    return strip


@torch.inference_mode()
def run_one_pair(row, xfeat, device, out_dir, args, measure=True):
    pair_id = str(row["pair_id"])
    t0 = time.perf_counter()
    img0 = resize_bgr(row["summer_path"], args.max_side)
    img1 = resize_bgr(row["winter_path"], args.max_side)
    ten0 = bgr_to_tensor(img0, device)
    ten1 = bgr_to_tensor(img1, device)
    sync_cuda(device)
    load_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    pts0, pts1 = get_star_matches(xfeat, img0, img1, ten0, ten1, args.top_k)
    sync_cuda(device)
    match_extract_ms = (time.perf_counter() - t1) * 1000.0
    t2 = time.perf_counter()
    H, inlier_mask, status = compute_homography(
        pts0, pts1, args.ransac_thresh, args.max_iters, args.confidence
    )
    homography_ms = (time.perf_counter() - t2) * 1000.0
    total_ms = load_ms + match_extract_ms + homography_ms
    if measure:
        match_vis = draw_matches(img0, img1, pts0, pts1, inlier_mask, pair_id)
        warp_vis = build_clean_warp(img0, img1, H, pair_id)
        cv2.imwrite(str(out_dir / "matches" / f"{pair_id}.jpg"), match_vis)
        cv2.imwrite(str(out_dir / "warps" / f"{pair_id}.jpg"), warp_vis)
    result = {
        "pair_id": pair_id,
        "summer_path": row["summer_path"],
        "winter_path": row["winter_path"],
        "status": status,
        "num_matches": int(len(pts0)),
        "num_inliers": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.mean()) if len(inlier_mask) else 0.0,
        "homography": json.dumps(H.tolist()),
        "load_ms": load_ms,
        "extract_ms": match_extract_ms,
        "match_ms": 0.0,
        "homography_ms": homography_ms,
        "total_ms": total_ms,
    }
    for col in row.index:
        if col.startswith("source_"):
            result[col] = row[col]
    return result


# Main pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", required=True)
    parser.add_argument("--out_dir", default="output_xfeat_star")
    parser.add_argument("--top_k", type=int, default=1200)
    parser.add_argument("--max_side", type=int, default=640)
    parser.add_argument("--ransac_thresh", type=float, default=4.0)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--warmup_iters", type=int, default=3)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "matches").mkdir(parents=True, exist_ok=True)
    (out_dir / "warps").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    xfeat = XFeat(top_k=args.top_k).eval()
    print("Running warmup...")
    for _ in range(args.warmup_iters):
        _ = run_one_pair(pairs.iloc[0], xfeat, device, out_dir, args, measure=False)
    results = []
    for _, row in pairs.iterrows():
        result = run_one_pair(row, xfeat, device, out_dir, args, measure=True)
        results.append(result)
        print(
            f"{result['pair_id']}: status={result['status']} | "
            f"matches={result['num_matches']} | "
            f"inliers={result['num_inliers']} | "
            f"ratio={result['inlier_ratio']:.3f} | "
            f"total_ms={result['total_ms']:.2f}"
        )
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "reports" / "latency_report.csv", index=False)
    summary = {
        "device": str(device),
        "features": "XFeat* semi-dense",
        "top_k": args.top_k,
        "max_side": args.max_side,
        "pairs_processed": int(len(df)),
        "mean_inlier_ratio": float(df["inlier_ratio"].mean()),
        "mean_total_ms": float(df["total_ms"].mean()),
        "mean_load_ms": float(df["load_ms"].mean()),
        "mean_extract_ms": float(df["extract_ms"].mean()),
        "mean_match_ms": float(df["match_ms"].mean()),
        "mean_homography_ms": float(df["homography_ms"].mean()),
        "statuses": df["status"].value_counts().to_dict(),
    }
    with open(out_dir / "reports" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
