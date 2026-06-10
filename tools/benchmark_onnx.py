from pathlib import Path

import json

import time

import cv2

import numpy as np

import pandas as pd

import onnxruntime as ort



ort.preload_dlls(directory="")



ONNX_MODEL = "deployment/xfeat_net_static_480x640.onnx"

PAIRS_CSV = "data/demo_pairs_snow_manual/pairs.csv"

OUT_JSON = "deployment/onnx_cuda_benchmark.json"



def load_480x640(path):

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:

        raise FileNotFoundError(path)

    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    arr = img.astype(np.float32) / 255.0

    arr = np.transpose(arr, (2, 0, 1))[None]

    return arr



def bench(session, arr, repeats=50, warmup=10):

    input_name = session.get_inputs()[0].name



    for _ in range(warmup):

        session.run(None, {input_name: arr})



    times = []

    for _ in range(repeats):

        t0 = time.perf_counter()

        session.run(None, {input_name: arr})

        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000.0)



    return times



pairs = pd.read_csv(PAIRS_CSV)



image_paths = []

for _, row in pairs.iterrows():

    image_paths.append(row["summer_path"])

    image_paths.append(row["winter_path"])



sessions = {

    "cpu": ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"]),

    "cuda": ort.InferenceSession(ONNX_MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]),

}



report = {

    "onnx_model": ONNX_MODEL,

    "input_shape": [1, 3, 480, 640],

    "available_providers": ort.get_available_providers(),

    "results": {},

}



for mode, session in sessions.items():

    print(f"\n=== {mode.upper()} ===")

    print("Actual providers:", session.get_providers())



    per_image = []

    for path in image_paths:

        arr = load_480x640(path)

        times = bench(session, arr)

        stats = {

            "image": path,

            "mean_ms": float(np.mean(times)),

            "median_ms": float(np.median(times)),

            "std_ms": float(np.std(times)),

            "min_ms": float(np.min(times)),

            "max_ms": float(np.max(times)),

        }

        per_image.append(stats)

        print(f"{Path(path).name}: mean={stats['mean_ms']:.3f} ms, median={stats['median_ms']:.3f} ms")



    report["results"][mode] = {

        "actual_providers": session.get_providers(),

        "mean_ms_across_images": float(np.mean([x["mean_ms"] for x in per_image])),

        "median_ms_across_images": float(np.mean([x["median_ms"] for x in per_image])),

        "per_image": per_image,

    }



cpu_mean = report["results"]["cpu"]["mean_ms_across_images"]

cuda_mean = report["results"]["cuda"]["mean_ms_across_images"]

report["speedup_cuda_vs_cpu"] = float(cpu_mean / cuda_mean)



Path(OUT_JSON).write_text(json.dumps(report, indent=2))



print("\nSaved:", OUT_JSON)

print("CPU mean ms:", cpu_mean)

print("CUDA mean ms:", cuda_mean)

print("CUDA speedup vs CPU:", report["speedup_cuda_vs_cpu"])
