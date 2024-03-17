This is the Readme file for training the models proposed in manuscript.
Date of last modification: 3.17.24

module load miniconda

module load cuDNN/8.0.5.39-CUDA-11.1.1

conda activate fmgene

If you do not have the packages, please do pip install them. Specifically, it is recommend to install pytorch, matplotlib, numpy, scipy, dicom, shap, pandas for running all functions on this source.

python main.py

The results will be printed on the test dataset. Please change the code if you'd like to see results on other splits (i.e. train, valid). The data were splitted as 6-2-2 as train-valid-test ratio, you can change it in the dataloader file, for other parameters, please see config.json.

If you would like to see other models (i.e. RNN, Gupta's, etc), please use other py files with 'main' as part of their names.

I. CNN

II. MLP

III. RNN

IV. Merged

V. Gupta's

Please submit an application to ADNI (https://adni.loni.usc.edu/) for access to the data
