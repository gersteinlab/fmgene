import os
import shutil
import torch
import matplotlib.pyplot as plt
import numpy as np
from networks import CNN_Wrapper  # Ensure this is the correct import
from utils import read_json  # Ensure this is the correct import
from scipy.ndimage import zoom  # For upsampling

def visualize_cnn_feature_maps(model, dataloader, input_shape=(64,64,24)):
    if not os.path.exists('./cnn_vis'):
        os.makedirs('./cnn_vis')

    # Assuming the first layer of your CNN is the convolution layer you want to visualize
    first_conv_layer = next(model.net.children())

    ad_count, nc_count = 0, 0  # Counters for 'AD' and 'NC' images

    for test_batch_data in dataloader:
        if ad_count >= 5 and nc_count >= 5:  # Stop after generating 5 pairs
            print('5 pairs ok.')
            break

        test_fmri, _, labels = test_batch_data
        test_fmri = test_fmri.to('cuda', dtype=torch.float)

        # Forward pass through the first conv layer
        with torch.no_grad():
            feature_maps = first_conv_layer(test_fmri)

        feature_maps = feature_maps.cpu().numpy()
        num_feature_maps = feature_maps.shape[1]  # Assuming shape is [batch_size, channels, depth, height, width]

        # Process feature maps for each label
        for label in np.unique(labels):
            if (label == 1 and ad_count >= 5) or (label == 0 and nc_count >= 5):
                continue  # Skip if already have 5 images for this label

            label_indices = np.where(labels == label)[0]
            label_feature_maps = feature_maps[label_indices]

            # Use max pooling and average over the batches for this label
            pooled_feature_maps = np.max(label_feature_maps, axis=(0, -1))  # Max over depth and batch

            fig, axs = plt.subplots(4, 4, figsize=(20, 20))  # Adjust for the number of feature maps

            for i in range(num_feature_maps):
                # Individual normalization
                feature_map = pooled_feature_maps[i]
                normalized_feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))

                # Upsampling to match input shape, if necessary
                zoom_factors = [input_shape[j]/feature_map.shape[j] for j in range(2)]
                upsampled_feature_map = zoom(normalized_feature_map, zoom_factors, order=1)

                ax = axs[i // 4, i % 4]
                im = ax.imshow(upsampled_feature_map, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Feature Map {i}')
                ax.axis('off')

            # Add a single colorbar for the entire figure
            fig.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical', fraction=0.01, pad=0.01)
            # Update title and save file name based on label
            label_title = 'AD' if label == 1 else 'NC'
            plt.suptitle(f'Max-Pooled and Normalized Feature Maps for {label_title}')
            plt.savefig(f'cnn_vis/pooled_normalized_feature_maps_{label_title}_{ad_count+1 if label == 1 else nc_count+1}.png')
            plt.close()

            # Update the count
            if label == 1:
                ad_count += 1
            else:
                nc_count += 1
   

def NeuralNet(config, train, wrapper):
    print('Dataset', config['type'])
    
    model_name = config['model_name'] + config['type']
    config['batch_size'] = 1

    for exp_idx in range(config['num_exps']):
        config['model_name'] = model_name + str(exp_idx)
        config['seed'] += exp_idx * 2
        net = wrapper(config)
        
        if train:
            net.train()
        else:
            net.load()

        net.net.eval()

        visualize_cnn_feature_maps(net, net.test_dataloader)

        break  # Remove this break if you want to iterate over all experiments
    return

def main():
    train = False
    
    if os.path.exists('./cnn_vis'):
        shutil.rmtree('./cnn_vis')
    os.makedirs('cnn_vis', exist_ok=True)

    # Load your configuration
    config_cnn = read_json('./config.json')['cnn_1']

    # Execute the NeuralNet function
    NeuralNet(config_cnn, train=train, wrapper=CNN_Wrapper)

if __name__ == '__main__':
    main()
