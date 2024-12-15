# Deep Learning Analysis of fMRI Data for Alzheimer's Disease Prediction

This repository contains the implementation of deep learning models for Alzheimer's Disease prediction using fMRI data, as described in our paper [Deep learning analysis of fMRI data for predicting Alzheimer's Disease: A focus on convolutional neural networks and model interpretability](https://doi.org/10.1371/journal.pone.0312848).

## Requirements

### Environment Setup
```bash
module load miniconda
module load cuDNN/8.0.5.39-CUDA-11.1.1
conda activate fmgene
```

### Required Packages
```bash
pip install torch matplotlib numpy scipy pydicom shap pandas
```

## Data Access

The data used in this study is from the Alzheimer's Disease Neuroimaging Initiative (ADNI). To access the data:
1. Submit an application at [ADNI](https://adni.loni.usc.edu/)
2. Complete ADNI's Data Use Agreement
3. Once approved, download the fMRI scans

## Models

This repository includes implementations of several models:

1. **CNN (Main Model)**
   - 3D Convolutional Neural Network
   - Achieves 92.8% accuracy on test set
   - Implementation in `main.py`

2. **Baseline Models**
   - RNN: `main_rnn.py`
   - MLP: `main_mlp.py`
   - Merged (CNN + genetic data): `main_merged.py`
   - Gupta's CNN implementation: `main_gupta.py`

## Usage

```bash
python main.py  # For the primary CNN model
```

### Configuration
- Model parameters can be modified in `config.json`
- Default train/validation/test split ratio: 60%/20%/20%
- Dataset split can be modified in the dataloader file

## Results

Results will be displayed for the test dataset by default. To view results for other splits:
- Modify the evaluation section in the respective main files
- Available splits: train, valid, test

## Last Updated
Dec. 15, 2024

## Citation
If you use this code, please cite our paper:
```bibtex
@article{zhou2024deep,
  title={Deep learning analysis of fMRI data for predicting Alzheimer’s Disease: A focus on convolutional neural networks and model interpretability},
  author={Zhou, Xiao and Kedia, Sanchita and Meng, Ran and Gerstein, Mark},
  journal={PloS one},
  volume={19},
  number={12},
  pages={e0312848},
  year={2024},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

## License
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Contact
xiaozhoucs16@gmail.com

Note: This implementation requires CUDA-capable hardware and appropriate CUDA/cuDNN installations.