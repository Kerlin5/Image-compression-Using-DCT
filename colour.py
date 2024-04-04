import numpy as np
import os
from skimage import io, color, exposure
from skimage.transform import resize

# Define the size of the block
BLOCK_SIZE = 8

# Define quantization matrix for higher quality compression
Q = np.array([
    [1, 1, 1, 2, 2, 4, 4, 4],
    [1, 1, 1, 2, 3, 4, 4, 4],
    [1, 1, 2, 2, 3, 4, 5, 5],
    [2, 2, 2, 3, 4, 5, 5, 5],
    [2, 2, 3, 4, 5, 5, 6, 6],
    [2, 3, 4, 5, 5, 6, 6, 6],
    [4, 4, 5, 5, 6, 6, 7, 7],
    [4, 4, 5, 6, 6, 7, 7, 7]
])

# Function to perform DCT on a block
def DCT(block):
    return np.round(np.dot(np.dot(DCT_matrix(), block), DCT_matrix().T))

# Function to generate DCT matrix
def DCT_matrix():
    N = BLOCK_SIZE
    dct_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_matrix[i][j] = 1 / np.sqrt(N)
            else:
                dct_matrix[i][j] = np.sqrt(2/N) * np.cos((2 * j + 1) * i * np.pi / (2 * N))
    return dct_matrix

# Function to quantize DCT coefficients
def quantize(block):
    return np.round(block / Q)

# Function to dequantize DCT coefficients
def dequantize(block):
    return block * Q

# Function to perform inverse DCT on a block
def inverse_DCT(block):
    return np.dot(np.dot(DCT_matrix().T, block), DCT_matrix())

# Get the image file path
image_path = input("Enter the path to the image file: ")

# Check if the file exists
if not os.path.isfile(image_path):
    print("Error: Image file not found.")
    exit(1)

# Load the image
image = io.imread(image_path)

# Get the dimensions of the original image
height, width, channels = image.shape

# Resize the image to half of its original size
resized_image = resize(image, (height // 2, width // 2), anti_aliasing=True)

# Get the dimensions of the resized image
resized_height, resized_width, resized_channels = resized_image.shape

# Calculate the number of blocks
num_blocks_height = resized_height // BLOCK_SIZE
num_blocks_width = resized_width // BLOCK_SIZE

# Process each block
compressed_image = np.zeros_like(resized_image)
for c in range(resized_channels):  # Loop through each color channel
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            # Extract the block
            block = resized_image[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, c]

            # Apply DCT
            dct_block = DCT(block)

            # Quantize DCT coefficients
            quantized_block = quantize(dct_block)

            # Dequantize DCT coefficients
            dequantized_block = dequantize(quantized_block)

            # Reconstruct the block using inverse DCT
            reconstructed_block = inverse_DCT(dequantized_block)

            # Update the compressed image with the reconstructed block
            compressed_image[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, c] = reconstructed_block

# Adjust the contrast of the compressed image
compressed_image = exposure.rescale_intensity(compressed_image, in_range='image', out_range=(0, 255))

# Prompt the user for the output file name with extension
output_file_name = input("Enter the name for the compressed image file (with extension, e.g., compressed_image.jpg): ")

# Save the compressed image without low contrast warning
io.imsave(output_file_name, compressed_image.astype(np.uint8), check_contrast=False)

print(f"Compression completed successfully. Compressed image saved as: {output_file_name}")