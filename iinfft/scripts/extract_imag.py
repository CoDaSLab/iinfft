import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from scipy.io import loadmat
import h5py

import matplotlib
matplotlib.use('TkAgg')

# Load the .mat file and extract the image
file_path = "iinfft/data/C22_1.mat"
mat_data = h5py.File(file_path,'r')
image = mat_data['M']  # Replace with your variable name

# Ensure the image is in a usable format
if isinstance(image, np.ndarray) and image.dtype == object:
    image = image[0, 0]  # Extract the numeric array if nested

image = np.array(image, dtype=float)

# Create a mask for the selected region
mask = np.zeros_like(image, dtype=bool)

# Callback for LassoSelector
def onselect(verts):
    global mask
    # Create a path from the vertices
    path = Path(verts)
    # Generate coordinates for all pixels
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))  # Note the reversed axes
    coords = np.column_stack((x.ravel(), y.ravel()))
    # Update the mask for points inside the path
    mask = path.contains_points(coords).reshape(image.shape)
    # Set selected region to zero
    image[mask] = np.nan
    # Update the display
    img_display.set_data(image)
    fig.canvas.draw_idle()

# Plot the image
fig, ax = plt.subplots()
img_display = ax.imshow(image, cmap='viridis', origin='lower')  # Ensure origin is 'lower'
ax.set_title("Draw a region to zero out (right-click or enter to finish)")

# Attach the LassoSelector
lasso = LassoSelector(ax, onselect)

# Show the interactive plot
plt.show()

# Save the modified image as a .npy file
output_path = "modified_image.npy"
np.save(output_path, image)
print(f"Modified image saved to {output_path}")
