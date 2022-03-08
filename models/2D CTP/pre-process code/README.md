# Pre-processing & convert CTP to masks

## Install pyelastix for registration
```
sudo apt update
sudo apt install libgl1-mesa-glx
sudo apt-get install elastix

pip install pip --upgrade
pip install opencv-contrib-python
pip install pyelastix
```

## Usage
- ncct_lesion_map.csv is the list of raw NCCT and raw CTP dicom pairs
- Preprocessing NCCT and CTP
    - Convert NCCT from dicom to png using 3 window sizes
    - Convert CTP raw images to masks (0: background, 255: stroke lesion/core+penumbra)
    - Perform registration between NCCT images and CTP masks
    - create_ncct_ctp_dataset.py
        - Input: ncct_lesion_map.csv
        - Output: images folder, masks folder
    - Run script
        ```python create_ncct_ctp_dataset.py```
        
- Remove tiny arrow if necessary
    - Because CTP masks possibly contain tiny arrows which was created by AutoMistar
    - An example about removing tiny arrows in notebook below
        - clean_arrows_in_masks.ipynb 