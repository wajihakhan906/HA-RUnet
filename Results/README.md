# Results

## Published results (BraTS-2020 test split)

As reported in *A Hybrid Attention-Based Residual Unet for Semantic Segmentation of Brain Tumor*
(Computers, Materials & Continua, 2023):

| Region | Dice | Sensitivity |
|---|---|---|
| Whole Tumor (WT) | 0.867 | 0.93 |
| Tumor Core (TC) | 0.813 | 0.88 |
| Enhancing Tumor (ET) | 0.787 | 0.83 |

HA-RUnet outperformed the ResUNet and AResUNet baselines while using fewer parameters.

## Reproducing

```bash
cd Code
python train.py --data ../Dataset/MICCAI_BraTS2020_TrainingData --out ../Results/run
python evaluate.py --data ../Dataset/MICCAI_BraTS2020_TrainingData --checkpoint ../Results/run/best_model.pt
python plot_history.py --history ../Results/run/history.json
```

`train.py` writes `history.json` and `best_model.pt` to `Results/run/`, and `evaluate.py` writes `metrics.json`.
