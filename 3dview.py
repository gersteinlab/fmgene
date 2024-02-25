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

        # Plot the fMRI data
        plot_fmri_data(fmri, exp_idx)
        
    return

def plot_fmri_data(fmri_tensor, exp_idx):
    # Convert the tensor to a numpy array and squeeze it if necessary
    fmri_data = fmri_tensor.squeeze().cpu().numpy()

    # Ensure the directory exists
    save_dir = './3d'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a NIfTI image (3D)
    affine = np.eye(4)  # Basic affine transformation matrix
    fmri_nifti = nib.Nifti1Image(fmri_data, affine)

    # Plot the 3D brain image without crosshairs
    display = plotting.plot_img(fmri_nifti, title=f'fMRI 3D View, Experiment {exp_idx}',
                                display_mode='ortho', cut_coords=None, draw_cross=False,
                                black_bg=True)

    # Remove axis labels, ticks, and coordinate labels
    for ax in display.axes.values():
        ax.ax.axis('off')  # Turn off axis
        ax.ax.set_xticklabels([])  # Remove x-axis labels
        ax.ax.set_yticklabels([])  # Remove y-axis labels
        ax.ax.set_zticklabels([])  # Remove z-axis labels (if present)

    # Save the figure
    display.savefig(f'{save_dir}/fmri_exp_{exp_idx}_3d.png')
    display.close()

def main():
    
    train = False
    if os.path.exists('./3d'):
        shutil.rmtree('./3d')
    os.makedirs('3d', exist_ok=True)

    # Load your configuration
    config_cnn = read_json('./config.json')['cnn_1']

    # Execute the NeuralNet function
    NeuralNet(config_cnn, train=train, wrapper=CNN_Wrapper)

if __name__ == '__main__':
    main()
