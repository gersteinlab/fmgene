import os
import shutil
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, datasets, image
from networks import CNN_Wrapper, MLP_Wrapper, RNN_Wrapper, M_Wrapper, CNN_paper, MLP_fMRI_Wrapper
from utils import read_json
from nilearn.datasets import load_mni152_brain_mask
from scipy.ndimage import zoom
from nilearn.plotting import cm


def NeuralNet(config, train, wrapper):
    print('Dataset', config['type'])
    
    config['model_name'] += config['type']
    model_name = config['model_name']
    config['batch_size'] = 1

    for exp_idx in range(config['num_exps']):
        config['model_name'] = model_name + str(exp_idx)
        config['seed'] += exp_idx*2
        net = wrapper(config)
        
        if train:
            net.train()
        else:
            net.load()

        net.net.eval()
        batch_data = next(iter(net.train_dataloader))
        fmri, gene, label = batch_data
        fmri = fmri.to('cuda', dtype=torch.float)
        background_data = fmri[:1]
        explainer = shap.DeepExplainer(net.net, background_data)

        test_batch_data = next(iter(net.test_dataloader))
        test_fmri, test_gene, test_label = test_batch_data
        test_fmri = test_fmri.to('cuda', dtype=torch.float)
        test_data_subset = test_fmri[:1]
        
        shap_values = explainer.shap_values(test_data_subset, check_additivity=False)
        shap_values_reshaped = np.array(shap_values)[0, 0, :, :, :].squeeze()  # Adjust indexes based on your data shape
        shap_values_scaled = shap_values_reshaped
        
        shap_threshold = 0.0000035  # Define a threshold
        shap_values_scaled = np.where(np.abs(shap_values_scaled) > shap_threshold, shap_values_scaled, 0)

        test_fmri_nifti = nib.Nifti1Image(test_fmri.cpu().numpy()[0, 0, :, :, :], affine=np.eye(4))
        fmri_data = test_fmri_nifti.get_fdata()
        fmri_data_min = fmri_data.min()
        fmri_data_max = fmri_data.max()
        normalized_fmri_data = (fmri_data - fmri_data_min) / (fmri_data_max - fmri_data_min)
        normalized_fmri_nifti = nib.Nifti1Image(normalized_fmri_data, affine=test_fmri_nifti.affine)

        shap_min = shap_values_scaled.min()
        shap_max = shap_values_scaled.max()
        normalized_shap_values = (shap_values_scaled - shap_min) / (shap_max - shap_min)

        normalized_shap_nifti = nib.Nifti1Image(normalized_shap_values, affine=test_fmri_nifti.affine)

        display = plotting.plot_stat_map(
            normalized_shap_nifti,  # SHAP values Nifti image
            bg_img=normalized_fmri_nifti,  # Background fMRI Nifti image
            display_mode='ortho',
            cmap='hot',  # Colormap for SHAP values
            colorbar=True,
            title="SHAP Values Overlay on Grayscale fMRI",
            threshold=0.5,  # Set a higher threshold to show only significant SHAP values as 'dots'
            alpha=1.0,  # Maximum value for transparency to make the dots solid
            black_bg=False,
        )
        display.savefig('plot/fmri_with_filtered_shap_overlay.png')
        display.close()

        fmri_data_array = normalized_shap_nifti.get_fdata()
        brain_mask_data_array = normalized_fmri_nifti.get_fdata()
        for i in range(11):  # This will create 11 steps from 0 to 1, inclusive
            fmri_factor = i / 10
            brain_mask_factor = 1 - fmri_factor
            combined_data_array = (fmri_data_array * fmri_factor) + (brain_mask_data_array * brain_mask_factor)
            combined_data_array[combined_data_array > 1] = 1
            combined_nifti = nib.Nifti1Image(combined_data_array, affine=test_fmri_nifti.affine)
            filename = f'plot/combined_fmri_{fmri_factor}_brain_mask_{brain_mask_factor}.png'

            display = plotting.plot_img(
                combined_nifti, 
                display_mode='ortho', 
                cmap='gray', 
                title=f"Combined fMRI (factor {fmri_factor}) and Brain Mask (factor {brain_mask_factor})",
                threshold='auto',
                black_bg=True
            )
            display.savefig(filename)
            display.close()

        break
    return

def main():
    
    train = False
    if os.path.exists('./plot'):
        shutil.rmtree('./plot')
    os.makedirs('plot', exist_ok=True)

    # Load your configuration
    config_cnn = read_json('./config.json')['cnn_1']

    # Execute the NeuralNet function
    NeuralNet(config_cnn, train=train, wrapper=CNN_Wrapper)

if __name__ == '__main__':
    main()
