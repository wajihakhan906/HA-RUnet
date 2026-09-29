# Dataset: BraTS-2020

HA-RUnet is trained and evaluated on the **Multimodal Brain Tumor Segmentation Challenge 2020 (BraTS-2020)**.
The data is not redistributed here. Register and download it from the official source:

- https://www.med.upenn.edu/cbica/brats2020/data.html

Each case contains four co-registered MRI modalities (FLAIR, T1, T1ce, T2; 240 × 240 × 155) and an expert
segmentation with labels **1** (necrotic / non-enhancing core), **2** (peritumoral edema) and **4** (enhancing tumor).

Place the extracted training folder here:

```
Dataset/
└── MICCAI_BraTS2020_TrainingData/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_flair.nii.gz
    │   ├── BraTS20_Training_001_t1.nii.gz
    │   ├── BraTS20_Training_001_t1ce.nii.gz
    │   ├── BraTS20_Training_001_t2.nii.gz
    │   └── BraTS20_Training_001_seg.nii.gz
    └── ...
```

The loader (`Code/dataset.py`) converts the labels into the three evaluated regions:

| Region | Labels |
|---|---|
| Whole Tumor (WT) | 1 + 2 + 4 |
| Tumor Core (TC) | 1 + 4 |
| Enhancing Tumor (ET) | 4 |
