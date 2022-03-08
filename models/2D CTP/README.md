# 2D CTP model

## 資料夾結構
```bash
/data
├── df_train.csv
├── df_val.csv
├── images
│   ├── A.png
│   ├── B.png
│   ├── C.png
│   ├── D.png
│   └── E.png
└── penucore_masks
    ├── A_core_pneu.png
    ├── B_core_pneu.png
    ├── C_core_pneu.png
    ├── D_core_pneu.png
    └── E_core_pneu.png
```
## Demo data
  
### df_train.csv、df_val.csv
```csv
case_id,image_id
6fc644827a,6fc644827a_1.2.840.113619.2.411.3.537526739.860.1585500646.720.23.png
6fc644827a,6fc644827a_1.2.840.113619.2.411.3.537526739.860.1585500646.724.25.png
6fc644827a,6fc644827a_1.2.840.113619.2.411.3.537526739.860.1585500646.724.27.png
22062ef1a9,22062ef1a9_1.2.840.113619.2.411.3.537526739.836.1584308117.584.18.png
```

### Image
- Pre-processing
  - windowing : [(40, 80), (80, 200), (600, 2800)] (WL、WW)
  - 3C => 1C : np.mean()
- Data Augmentation
  - HorizontalFlip()
  - VerticalFlip()
  - Rotate()
  - RandomContrast(limit=0.2, prob=0.5)
  - RandomHueSaturationValue()
  - Normalize()
### MASK
- Background = 0
- 缺血、penumbra 跟 core = 255