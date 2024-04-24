# StrainNet
**Improved Myocardial Strain Analysis of Cine MRI by Deep Learning from DENSE** | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10316292/pdf/ryct.220196.pdf)

StrainNet analysis of routine cardiac cine MRI provides clinical convenience similar to feature tracking and better agreement with displacement encoding with stimulated echoes for both global and segmental circumferential strain.

**Reference**:
Wang, Yu, et al. "StrainNet: improved myocardial strain analysis of cine MRI by deep learning from DENSE." Radiology: Cardiothoracic Imaging 5.3 (2023): e220196.

---

# StrainNet-Transformer
**Improved Strain Analysis of Cine MRI with Long-Range Spatiotemporal Relationship Learning**

StrainNet-Transformer is a transformer-based DL framework that combines the attention mechanisms for long-term dependencies and convolutions for local dependencies, capturing both global and local data for improved intramyocardial motion estimation from contour motion. 

*Updated: April 2024*
<br/><br/>

## Getting Started
Current version is implemented with Python 3.9 and PyTorch 1.10.0. 

Other packages and versions are listed in `requirements.txt`.
<br/><br/>

## Data
- Testing and training data are located at `./data/test_cine/` and `./data/train/` folders. One example testing and one example traning dataset are provided for reference.
- For testing, `input` is a series of binarized images of LV myocardium, with size of [1, N<sub>x</sub>, N<sub>y</sub>, N<sub>t</sub>]. 
- For training, additional ground truth displacement `label` is needed, with size of [2, N<sub>x</sub>, N<sub>y</sub>, N<sub>t</sub>].
<br/><br/>

## Pre-trained Model
A pre-trained model is available [here](https://www.dropbox.com/scl/fi/2ezqjv04l6ulob1ku1wzm/model_best.pth?rlkey=go43i5sfc8f8b2zage9em4ikf&dl=0). This model should be put in `./saved/models/StrainNet/0422_000912/` folder.
<br/><br/>

## Testing
A jupyter notebook demo [`StrainNet_demo.ipynb`](StrainNet_demo.ipynb) shows the inference of StrainNet, and the testing example results are embedded.

Another way to test is to run `test_cine.py` and the corresponding testing configuration is implemented at `config_test_unet_cine.json`.
<br/><br/>

## Training
Run `train.py` and the corresponding training configuration is implemented at `config_train_unet.json`.
<br/><br/>

## Contact
Yu Wang - yw8za@virginia.edu
