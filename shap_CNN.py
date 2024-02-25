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

def adjust_contrast(image_data, min_percentile=55, max_percentile=100):
    """ Adjust the contrast of an image based on percentile intensities. """
    min_val = np.percentile(image_data, min_percentile)
    max_val = np.percentile(image_data, max_percentile)
    contrast_adjusted = np.clip((image_data - min_val) / (max_val - min_val), 0, 1)
    return contrast_adjusted

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

        for label_val in [0, 1]:
            count = 0
            for test_batch_data in net.test_dataloader:
                test_fmri, test_gene, test_label = test_batch_data
                if test_label.item() != label_val:
                    continue
                    
                test_fmri, test_gene, test_label = test_batch_data
                test_fmri = test_fmri.to('cuda', dtype=torch.float)
                test_fmri_nifti = nib.Nifti1Image(test_fmri.cpu().numpy()[0, 0, :, :, :], affine=np.eye(4))
                fmri_data = test_fmri_nifti.get_fdata()
                fmri_data_min = fmri_data.min()
                fmri_data_max = fmri_data.max()
                normalized_fmri_data = (fmri_data - fmri_data_min) / (fmri_data_max - fmri_data_min)
                fmri_contrast_adjusted = normalized_fmri_data
                # fmri_contrast_adjusted = adjust_contrast(normalized_fmri_data)
                normalized_fmri_nifti = nib.Nifti1Image(fmri_contrast_adjusted, affine=test_fmri_nifti.affine)
                
                test_data_subset = test_fmri[:1]
                shap_values = explainer.shap_values(test_data_subset, check_additivity=False)
                shap_values_reshaped = np.array(shap_values)[0, 0, :, :, :].squeeze()  # Adjust indexes based on your data shape
                shap_values_flat = shap_values_reshaped.flatten()

                shap_min = shap_values_reshaped.min()
                shap_max = shap_values_reshaped.max()
                normalized_shap_values = (shap_values_reshaped - shap_min) / (shap_max - shap_min)

                adaptive_threshold = np.percentile(np.abs(normalized_shap_values), 97)
                shap_values_scaled = np.where(np.abs(normalized_shap_values) > adaptive_threshold, normalized_shap_values, 0)

                normalized_shap_nifti = nib.Nifti1Image(shap_values_scaled, affine=test_fmri_nifti.affine)

                display = plotting.plot_stat_map(
                    normalized_shap_nifti,  # SHAP values Nifti image
                    bg_img=normalized_fmri_nifti,  # Background fMRI Nifti image
                    display_mode='ortho',
                    cmap='hot',  # Colormap for SHAP values
                    colorbar=True,
                    title="SHAP Values Overlay on Grayscale fMRI",
                    threshold=adaptive_threshold*0.9,  # Set a higher threshold to show only significant SHAP values as 'dots'
                    alpha=0.6,  # Maximum value for transparency to make the dots solid
                    black_bg=False,
                )
                display.savefig(f'plot/fmri_with_filtered_shap_overlay_label_{label_val}_instance_{count}.png')
                display.savefig(f'plot/fmri_with_filtered_shap_overlay_label_{label_val}_instance_{count}.tiff')
                display.close()

                mni152_template = datasets.load_mni152_template()
                display_shap = plotting.plot_stat_map(
                    normalized_shap_nifti,
                    display_mode='z',  # Adjust this to choose the axis for MIP ('x', 'y', or 'z')
                    cmap='hot',
                    title="SHAP Values MIP",
                    colorbar=True,
                    threshold='auto',  # Automatically choose a threshold to highlight significant SHAP values
                    black_bg=True,
                )
                display_shap.savefig(f'plot/shap_values_mip_label_{label_val}_instance_{count}.png')
                display_shap.savefig(f'plot/shap_values_mip_label_{label_val}_instance_{count}.tiff')
                display_shap.close()

                # fMRI MIP using the MNI152 template or the adjusted fMRI image
                display_fmri = plotting.plot_img(
                    mni152_template,  # Replace with `normalized_fmri_nifti` if using individual fMRI images
                    display_mode='z',  # Use the same axis as for SHAP MIP for consistency
                    cmap='gray',
                    title="fMRI MIP",
                    black_bg=True,
                )
                display_fmri.savefig(f'plot/fmri_mip_label_{label_val}_instance_{count}.png')
                display_fmri.savefig(f'plot/fmri_mip_label_{label_val}_instance_{count}.tiff')
                display_fmri.close()
                count += 1
                if count >= 5:
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
