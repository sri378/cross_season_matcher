import argparse

from pathlib import Path

import cv2

import numpy as np

import pandas as pd



def load_resized(path, max_side):

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:

        raise FileNotFoundError(path)



    h, w = img.shape[:2]

    scale = min(1.0, max_side / max(h, w))

    if scale < 1.0:

        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

    return img



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pair_id", required=True)

    ap.add_argument("--pairs_csv", default="data/demo_pairs_snow_manual/pairs.csv")

    ap.add_argument("--out_dir", default="annotations/final_snow")

    ap.add_argument("--max_side", type=int, default=640)

    args = ap.parse_args()



    df = pd.read_csv(args.pairs_csv)

    row = df[df["pair_id"] == args.pair_id].iloc[0]



    src = load_resized(row["summer_path"], args.max_side)

    tgt = load_resized(row["winter_path"], args.max_side)



    gap = 35

    label_h = 90

    H = max(src.shape[0], tgt.shape[0])

    W = src.shape[1] + gap + tgt.shape[1]



    base = np.ones((H + label_h, W, 3), dtype=np.uint8) * 255

    base[:src.shape[0], :src.shape[1]] = src

    base[:tgt.shape[0], src.shape[1] + gap:src.shape[1] + gap + tgt.shape[1]] = tgt



    src_pts = []

    tgt_pts = []



    def redraw():

        canvas = base.copy()



        cv2.putText(canvas, "LEFT: Spring/reference source", (20, H + 30),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.putText(canvas, "RIGHT: Winter/snow target", (src.shape[1] + gap + 20, H + 30),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.putText(canvas, "Click source point then matching target point. s=save, u=undo, q=quit",

                    (20, H + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)



        for i, p in enumerate(src_pts):

            p = tuple(map(int, p))

            cv2.circle(canvas, p, 5, (0, 255, 0), -1)

            cv2.putText(canvas, str(i + 1), (p[0] + 6, p[1] - 6),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)



        for i, p in enumerate(tgt_pts):

            p2 = (int(p[0] + src.shape[1] + gap), int(p[1]))

            cv2.circle(canvas, p2, 5, (0, 255, 0), -1)

            cv2.putText(canvas, str(i + 1), (p2[0] + 6, p2[1] - 6),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)



        return canvas



    def on_mouse(event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONDOWN:

            return



        if x < src.shape[1] and y < src.shape[0]:

            src_pts.append([x, y])

            print(f"source {len(src_pts)}: {x}, {y}")



        elif x > src.shape[1] + gap and y < tgt.shape[0]:

            tx = x - src.shape[1] - gap

            tgt_pts.append([tx, y])

            print(f"target {len(tgt_pts)}: {tx}, {y}")



    cv2.namedWindow(args.pair_id, cv2.WINDOW_NORMAL)

    cv2.setMouseCallback(args.pair_id, on_mouse)



    while True:

        cv2.imshow(args.pair_id, redraw())

        key = cv2.waitKey(30) & 0xFF



        if key == ord("u"):

            if len(tgt_pts) >= len(src_pts) and tgt_pts:

                tgt_pts.pop()

            elif src_pts:

                src_pts.pop()

            print("undo")



        elif key == ord("s"):

            n = min(len(src_pts), len(tgt_pts))

            if n < 6:

                print("Need at least 6 accurate corresponding points. 8-12 is better.")

                continue



            rows = []

            for i in range(n):

                rows.append({

                    "sx": src_pts[i][0],

                    "sy": src_pts[i][1],

                    "tx": tgt_pts[i][0],

                    "ty": tgt_pts[i][1],

                })



            out_dir = Path(args.out_dir)

            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{args.pair_id}_points.csv"

            pd.DataFrame(rows).to_csv(out_path, index=False)

            print(f"saved {out_path}")

            break



        elif key == ord("q"):

            print("quit without saving")

            break



    cv2.destroyAllWindows()



if __name__ == "__main__":

    main()
