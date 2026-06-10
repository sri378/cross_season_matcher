from pathlib import Path

import ast

import json



import cv2

import numpy as np

import pandas as pd



PAIRS_CSV = Path("data/demo_pairs_snow_manual/pairs.csv")

LATENCY_CSV = Path("output_final_snow_manual_r5/reports/latency_report.csv")

ANN_DIR = Path("annotations/final_snow")

OUT_DIR = Path("output_final_snow_manual_r5/reports")

MAX_SIDE = 640





def resized_shape(path):

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:

        raise FileNotFoundError(path)



    h, w = img.shape[:2]

    scale = min(1.0, MAX_SIDE / max(h, w))

    new_w = int(round(w * scale))

    new_h = int(round(h * scale))

    return new_w, new_h





def transform_points(H, pts):

    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)

    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    return out





def four_corner_error(H_est, H_ref, w, h):

    corners = np.array(

        [

            [0, 0],

            [w - 1, 0],

            [w - 1, h - 1],

            [0, h - 1],

        ],

        dtype=np.float32,

    )



    est_corners = transform_points(H_est, corners)

    ref_corners = transform_points(H_ref, corners)



    errors = np.linalg.norm(est_corners - ref_corners, axis=1)

    return float(errors.mean()), errors.tolist()





def main():

    pairs = pd.read_csv(PAIRS_CSV)

    latency = pd.read_csv(LATENCY_CSV)



    rows = []



    for _, pair in pairs.iterrows():

        pair_id = pair["pair_id"]

        ann_path = ANN_DIR / f"{pair_id}_points.csv"



        if not ann_path.exists():

            rows.append({

                "pair_id": pair_id,

                "status": "missing_annotation",

            })

            continue



        ann = pd.read_csv(ann_path)



        if len(ann) < 4:

            rows.append({

                "pair_id": pair_id,

                "status": "not_enough_points",

                "manual_points": len(ann),

            })

            continue



        src_pts = ann[["sx", "sy"]].to_numpy(np.float32)

        tgt_pts = ann[["tx", "ty"]].to_numpy(np.float32)



        H_manual, manual_mask = cv2.findHomography(

            src_pts,

            tgt_pts,

            method=cv2.USAC_MAGSAC,

            ransacReprojThreshold=3.0,

            maxIters=10000,

            confidence=0.999,

        )



        if H_manual is None:

            rows.append({

                "pair_id": pair_id,

                "status": "manual_homography_failed",

                "manual_points": len(ann),

            })

            continue



        lat_row = latency[latency["pair_id"] == pair_id].iloc[0]

        H_est = np.array(ast.literal_eval(lat_row["homography"]), dtype=np.float64)



        projected = transform_points(H_est, src_pts)

        point_errors = np.linalg.norm(projected - tgt_pts, axis=1)



        w, h = resized_shape(pair["summer_path"])



        mace, corner_errors = four_corner_error(H_est, H_manual, w, h)



        rows.append({

            "pair_id": pair_id,

            "status": "ok",

            "manual_points": len(ann),

            "manual_inliers": int(manual_mask.sum()) if manual_mask is not None else None,

            "manual_reproj_mean_px": float(point_errors.mean()),

            "manual_reproj_median_px": float(np.median(point_errors)),

            "manual_reproj_rmse_px": float(np.sqrt(np.mean(point_errors ** 2))),

            "four_corner_mace_px": mace,

            "corner_1_px": float(corner_errors[0]),

            "corner_2_px": float(corner_errors[1]),

            "corner_3_px": float(corner_errors[2]),

            "corner_4_px": float(corner_errors[3]),

            "passes_5px_mace": bool(mace < 5.0),

        })



    out = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)



    out_csv = OUT_DIR / "corner_error_report.csv"

    out.to_csv(out_csv, index=False)



    ok = out[out["status"] == "ok"].copy()



    summary = {

        "pairs_evaluated": int(len(ok)),

        "mean_four_corner_mace_px": float(ok["four_corner_mace_px"].mean()) if len(ok) else None,

        "max_four_corner_mace_px": float(ok["four_corner_mace_px"].max()) if len(ok) else None,

        "all_pairs_under_5px_mace": bool(ok["passes_5px_mace"].all()) if len(ok) else False,

        "mean_manual_reprojection_error_px": float(ok["manual_reproj_mean_px"].mean()) if len(ok) else None,

    }



    out_json = OUT_DIR / "corner_error_summary.json"

    out_json.write_text(json.dumps(summary, indent=2))



    print("\nSaved:", out_csv)

    print("Saved:", out_json)

    print("\nPer-pair corner error:")

    print(out.to_string(index=False))

    print("\nSummary:")

    print(json.dumps(summary, indent=2))





if __name__ == "__main__":

    main()
