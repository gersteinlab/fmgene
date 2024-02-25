import os
import numpy as np
from nilearn import datasets, image, plotting
import matplotlib.pyplot as plt

# Fetch the atlas and MRI template
atlas_data = datasets.fetch_atlas_aal()
mri_img = datasets.load_mni152_template()

# Load atlas image and get data as a numpy array
atlas_img = image.load_img(atlas_data.maps)
atlas_data_array = atlas_img.get_fdata()

# Find region IDs for the hippocampus
hippocampus_id_L = atlas_data.indices[atlas_data.labels.index("Hippocampus_L")]
hippocampus_id_R = atlas_data.indices[atlas_data.labels.index("Hippocampus_R")]
hippocampus_ids = [int(hippocampus_id_L), int(hippocampus_id_R)]

# Create a mask for the hippocampus
hippocampus_mask_array = np.isin(atlas_data_array, hippocampus_ids)

# Check if the mask is not empty
if np.sum(hippocampus_mask_array) == 0:
    raise ValueError("Hippocampus mask is empty. Check the region IDs.")
else:
    print("Hippocampus mask created successfully.")

# Convert the mask back to an image
hippocampus_mask_img = image.new_img_like(atlas_img, hippocampus_mask_array)

# Overlay the hippocampus mask on the MRI image
display = plotting.plot_anat(mri_img, title="Hippocampus Highlighted", cut_coords=(0, -30, 0))
display.add_contours(hippocampus_mask_img, levels=[0.5], colors='r')

# Ensure the '3d' directory exists
output_dir = '3d'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the result as a PNG file in the '3d' folder
output_file = os.path.join(output_dir, 'hippocampus_3d_plot.png')
plt.savefig(output_file)
print(f"Saved hippocampus visualization to {output_file}")



view = plotting.view_img(hippocampus_mask_img, bg_img=mri_img, title="3D View of the Hippocampus")
output_html = os.path.join(output_dir, 'hippocampus_3d_view.html')
view.save_as_html(output_html)
print(f"3D view of the hippocampus saved as HTML at {output_html}")