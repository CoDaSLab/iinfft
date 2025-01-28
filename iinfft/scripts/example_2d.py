import numpy as np
import matplotlib.pyplot as plt
from iinfft.iinfft import *
plt.style.use('tableau-colorblind10')
import sym_matrix

import matplotlib
matplotlib.use('TkAgg')

from scipy.io import loadmat

image = np.load('iinfft/data/modified_image.npy')


# Forward transform
N = 64
w = sobk(N,1,2,1e-2)
transformed_data, mtot = infft_2d(image, N,w=w)

# Adjoint transform (reconstruction)
reconstructed_data = adjoint_transform_2d(
    transformed_data,
    mtot,
    data_shape=image.shape
)

# Replace NaNs in the original image with interpolated values
interpolated_image = np.copy(image)  # Copy the original image
nan_mask = np.isnan(image)  # Identify NaN locations
interpolated_image[nan_mask] = reconstructed_data[nan_mask]  # Replace NaNs with reconstructed values

# Visualize results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Data")
plt.imshow(image, aspect='auto', cmap='viridis')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Reconstructed Data")
plt.imshow(reconstructed_data, aspect='auto', cmap='viridis')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Original with Interpolated NaNs")
plt.imshow(interpolated_image, aspect='auto', cmap='viridis')
plt.colorbar()

plt.tight_layout()
plt.show()