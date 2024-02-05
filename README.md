# icml2024


## Installation 
```bash
conda create -n vltrojan python=3.9 -y
conda activate vltrojan
pip install -r train/requirements.txt
pip install -r eval/requirements-eval.txt
mkdir train/checkpoints_remote
cd eval && pip install -v -e . 
```

## Data Preparation
### Training data
For more details on navigating the dataset, please refer to the **Official** [MIMIC-IT Dataset README](train/mimic-it/README.md).
1. Download the Official MIMIC-IT `instructions.json` and `train.json` files, [OneDrive folder](https://entuedu-my.sharepoint.com/:f:/g/personal/libo0013_e_ntu_edu_sg/Eo9bgNV5cjtEswfA-HfjNNABiKsjDzSWAl5QYAlRZPiuZA?e=M9isDT).
2. Request and Download the Official MIMIC-IT `XXX.json` raw images, [google form](https://docs.google.com/forms/d/e/1FAIpQLSfZOxo8ML5wwGWJzGuIG4qlcj2rsw4sRjT929V-fBWVU7SIcQ/viewform)
3. Directory 
``` 
train
└── mimic-it/  
    ├── CGD   
    │    ├── CGD_instructions.json  
    │    ├── CGD.json  
    │    ├── CGD_train.json  
    ├── convert-it  
    ├── DC   
    ├── E4D   
    ├── LA   
    │    ├── LA_instructions.json  
    │    ├── LADD.json  
    │    ├── LADD_train.json 
    │    ├── ......
    ├── README.md  
    ├── SD  
    ├── SN   
    ├── syphus  
    ├── TVC     
    ├── VST   
```
### Testing data
Following the **Official** Flamingo [Evaluation Dataset README](eval/open_flamingo/eval/README.md).
```
eval
└── data/  
    ├── flickr30k
    ├── hatefulmemes
    ├── imagenet
    ├── mscoco_karpathy
    ├── okvqa
    ├── README.md
    ├── textvqa
    ├── vizwiz
    └── vqav2
```

## Train
```bash
cd train && sh ru_train.sh
```

## Test
```bash
cd eval && sh ru_eval.sh
```

## Trigger Generation
### Image Trigger
```bash
cd train/pipeline/utils/backdoor/trigger_generate/ && python img_trigger.py
# using the convert.py to convert generated .pt to .png
```
### Text Trigger
```bash
cd train/pipeline/utils/backdoor/trigger_generate/ && python text_trigger.py
```