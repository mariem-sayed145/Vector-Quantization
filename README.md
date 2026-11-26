# Vector Quantization (VQ) — RGB Image Compression

## Overview

This project implements **Vector Quantization (VQ)** for image compression and decompression using Python.

The main goal is to reduce image storage size by representing image blocks using a limited codebook of representative vectors.

The project supports **RGB color images**, preserving all three color channels (Red, Green, and Blue) throughout the compression and decompression processes.

## Features

- Load RGB images.
- Divide images into non-overlapping blocks (vectors).
- Handle images whose dimensions are not divisible by the selected block size using padding.
- Generate a codebook containing representative vectors.
- Compress images by replacing each block with its nearest codebook index.
- Decompress images using the saved codebook and encoded indices.
- Reconstruct the compressed image.
- Calculate the compression ratio.
- Compare the original and reconstructed images.

## Technologies Used

- Python 3.10+
- NumPy
- Pillow
- Pytest

## Project Structure

```text
Vector-Quantization/
│
├── data/
│   ├── input/
│   └── output/
│
├── src/
│   ├── __init__.py
│   ├── image_processing.py
│   ├── codebook.py
│   ├── compression.py
│   ├── decompression.py
│   └── metrics.py
│
├── tests/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: The project structure may be updated during implementation.*

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mariem-sayed145/Vector-Quantization.git
```

### 2. Navigate to the Project Directory

```bash
cd Vector-Quantization
```

### 3. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## How It Works

### Compression

1. Load the input RGB image.
2. Divide the image into non-overlapping blocks.
3. Apply padding if necessary.
4. Convert each block into a vector.
5. Generate a codebook with the required number of codewords.
6. Find the nearest codeword for each image vector.
7. Store the encoded indices, codebook, and required image metadata.

### Decompression

1. Load the encoded indices and codebook.
2. Replace each index with its corresponding codeword.
3. Reconstruct the image blocks.
4. Combine the blocks to rebuild the image.
5. Remove the padding to restore the original dimensions.
6. Save the reconstructed RGB image.

## RGB Image Support

The implementation processes color images using three channels:

- Red (R)
- Green (G)
- Blue (B)

All three channels are preserved during block extraction, codebook generation, compression, and decompression.

For example, a 2 × 2 RGB block contains 12 values:

`2 × 2 × 3 = 12`

## Compression Ratio

The compression ratio measures how much the image size is reduced.

```text
Compression Ratio = Original Size / Compressed Size
```

The compressed size should account for:

- Encoded indices.
- Codebook storage.
- Required metadata.

A higher compression ratio indicates better compression.

## Running the Project

After completing the implementation and integration:

```bash
python main.py
```

## Running Tests

To execute the unit tests:

```bash
python -m pytest -v
```

## Team Contributions

The project is divided into the following main tasks:

- **Image Processing & Codebook Generation:** Image loading, RGB processing, padding, block extraction, and codebook generation.
- **Compression:** Encoding image vectors into codebook indices and storing the compressed representation.
- **Decompression:** Reconstructing the image using the codebook and encoded indices.
- **Testing & Integration:** Testing the implementation and integrating all project components.

## Important Notes

- RGB images must be supported throughout the entire pipeline.
- Padding must be handled correctly.
- Compression and decompression must use a consistent data format.
- The original image dimensions must be preserved in the metadata.
- The compression ratio must include the codebook size.
