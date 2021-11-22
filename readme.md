# Learnable Locality-Sensitive Hashing for Video Anomaly Detection

### This repository is the official implementation of *Learnable Locality-Sensitive Hashing for Video Anomaly Detection*.

## Dependencies
- Hardware
  - CPU: 48 cores
  - RAM: 384 GB
  - Disk: 3 TB
  - GPU: NVIDIA GeForce 2080 Ti * 4
- Software
  - python 3.8.8 (in Anaconda)
  - cudatoolkit 11.1.1
  - pytorch 1.8.1
  - torchvision 0.9.1
  - numpy 1.19.2
  - scikit-learn 0.24.2
  - scipy 1.7.1
  - ffmpeg 4.3.2
  - opencv 4.5.3.56
  - slowfast 1.0 (https://github.com/facebookresearch/SlowFast)

## Datasets
- Avenue: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
- ShanghaiTech: https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection
- Corridor: https://rodrigues-royston.github.io/Multi-timescale_Trajectory_Prediction

## Usage
1. Extract features for the three datasets: `0-FeatureExtraction`.
2. (Light-)LSH|LLSH: `1-LLSH`.
2. KNN: `2-KNN`.
3. K-means: `3-KMeans`.

**Please refer to the subdirectories for more details.**


## License
TODO

## Related Repositories
- MoCo: https://github.com/facebookresearch/moco
- SlowFast: https://github.com/facebookresearch/SlowFast