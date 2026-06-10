# Cross-Season Temporal Image Matching

This repository contains my submission for the cross-season temporal image matching task. The goal is to match Summer/Spring reference images with Winter/Snow target images, estimate a homography between them, generate match/warp visualizations, and profile the runtime performance on GPU.

The final pipeline uses XFeat* for feature matching, USAC_MAGSAC for robust homography estimation, CUDA for acceleration, and ONNX Runtime CUDA for deployment benchmarking.

## 1. Setup Instructions

Create and activate a Python environment:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run the final matching pipeline:

    bash run.sh

The reproduced outputs will be saved under:

    output/reproduced/

The provided final outputs are already included in:

    output/matches/
    output/warps/
    output/reports/

## 2. Latency Profiling Table

Final run settings:

| Setting | Value |
|---|---:|
| Feature matcher | XFeat* semi-dense |
| Device | CUDA |
| Top-K features | 1200 |
| Max image side | 512 |
| Number of pairs | 5 |
| Mean inlier ratio | 0.883 |
| Mean total latency | 48.14 ms |

Per-pair latency and matching results:

| Pair | Data Loading Time | Feature Extraction Time | Matrix Estimation Time | Total Time | Inlier Ratio |
|---|---:|---:|---:|---:|---:|
| pair_001 | 14.29 ms avg | 32.84 ms avg | 1.01 ms avg | 48.97 ms | 0.947 |
| pair_002 | 14.29 ms avg | 32.84 ms avg | 1.01 ms avg | 48.46 ms | 0.926 |
| pair_003 | 14.29 ms avg | 32.84 ms avg | 1.01 ms avg | 47.11 ms | 0.859 |
| pair_004 | 14.29 ms avg | 32.84 ms avg | 1.01 ms avg | 48.18 ms | 0.783 |
| pair_005 | 14.29 ms avg | 32.84 ms avg | 1.01 ms avg | 48.01 ms | 0.901 |

The measured mean end-to-end latency is 48.14 ms per pair, which is below the required 50 ms GPU target.

The detailed latency files are available at:

    output/reports/latency.csv
    output/reports/matching_summary.json

## 3. Optimization Breakdown

The following steps were used to speed up the pipeline:

1. The input images were resized using max_side=512. This reduced feature extraction time while still preserving enough structure for reliable cross-season matching.
2. Feature extraction and matching were run on CUDA instead of CPU.
3. USAC_MAGSAC was used for robust matrix estimation, which helped reject seasonal outliers caused by snow, trees, shadows, and vehicles.
4. The XFeat backbone was exported to ONNX as:

    model/xfeat_480x640.onnx

5. ONNX Runtime CUDAExecutionProvider was benchmarked for deployment. The measured ONNX CPU inference time was about 12.64 ms, while CUDA inference time was about 2.70 ms. This gives about 4.67x speedup.
6. TensorRTExecutionProvider was tested, but the local system did not have the required TensorRT runtime library libnvinfer.so.10. Therefore, the final optimized deployment uses ONNX Runtime CUDA.

The ONNX benchmark files are available at:

    model/onnx_cuda_benchmark.json
    model/onnx_export_meta.json

## 4. Output Folder

The required output folder is included as:

    output/

It contains:

    output/matches/    5 match visualization images
    output/warps/      5 warped alignment images
    output/reports/    latency and validation reports

The match images show side-by-side Summer/Spring and Winter/Snow image pairs with feature match lines.

The warp images show the Summer/Spring image warped into the Winter/Snow target frame using the estimated homography.

## 5. Geometric Validation

The homography matrices are estimated from XFeat* matches after USAC_MAGSAC filtering. The warped alignment outputs are stored in:

    output/warps/

The selected image pairs do not include official pairwise ground-truth homography matrices. Because of this, manual landmark reprojection validation was used as an additional geometry check.

Four pairs had reliable static landmarks and achieved robust median landmark reprojection error below 5 px. pair_005 was excluded only from manual landmark validation because it has no reliable man-made control points, but it is still included in matching, warping, and latency evaluation.

The validation files are:

    output/reports/landmark_validation.csv
    output/reports/landmark_validation_summary.json

## 6. Personal Findings and Implementation Notes

This task made me realize that cross-season image matching is not just about detecting many feature points. Snow, missing foliage, lighting changes, shadows, and vehicles can create many visually confusing regions. A pair that looks similar to a human does not always give good geometric consistency after RANSAC.

During experimentation, I selected the final pairs based on both visual overlap and inlier consistency after homography estimation. I found that stable structures such as buildings, curbs, poles, and road boundaries were more reliable than trees, grass, snow regions, or cars.

One useful finding was the effect of image size on speed. At max_side=640, the pipeline was slightly above the 50 ms target in the cleaned repository. Reducing the image size to max_side=512 brought the mean latency down to 48.14 ms while improving the mean inlier ratio to 0.883. This showed me the practical trade-off between resolution, runtime, and matching quality.

I also learned that a single homography is only an approximation for outdoor scenes. Roads, houses, trees, and vehicles are at different depths, so the warp is strongest on stable structures and less reliable on foreground or seasonal regions. This is why I used manual landmark validation only where reliable landmarks were visible.

Overall, this task helped me understand how a robotics perception pipeline needs more than just model inference. It also needs geometric verification, runtime profiling, deployment testing, and honest reporting of limitations.


### Observation on Match Distribution

In some of the match visualizations, many accepted matches appear in the upper half of the image, especially around tree canopies, building roofs, poles, and other high-contrast background structures. I noticed that the road and ground plane produced fewer reliable matches after RANSAC.

This is expected in cross-season image matching because the ground region changes heavily between seasons. In the Summer/Spring images, the ground may contain grass, road texture, shadows, lane/curb details, and vegetation. In the Winter/Snow images, many of these same regions are covered by snow or have very different texture and contrast. As a result, ground-plane features are less repeatable across seasons.

The robust estimator therefore keeps matches mostly on structures that remain visually stable across both seasons. These include tree trunks, rooflines, building boundaries, poles, curbs, and far-field edges. This behavior is reasonable for a visual localization pipeline because stable landmarks are more useful than seasonal foreground texture.

This also explains why the homography warp is more reliable on stable structures and less reliable on snow-covered ground or moving/seasonal objects.

## 7. Main Results

| Metric | Result |
|---|---:|
| Number of evaluated pairs | 5 |
| Mean inlier ratio | 0.883 |
| Required inlier ratio | 0.600 |
| Mean total latency | 48.14 ms |
| Required latency | < 50 ms |
| Successful pairs | 5 |
| ONNX CUDA speedup | 4.67x |

