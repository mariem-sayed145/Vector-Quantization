import numpy as np
from PIL import Image
from pathlib import Path


def load_image(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_array = np.array(image, dtype=np.uint8)

    except Exception as error:
        raise ValueError(f"Unable to load image: {error}") from error

    return image_array


def pad_image(image: np.ndarray, block_size: tuple[int, int]):

    block_height, block_width = block_size

    if block_height <= 0 or block_width <= 0:
        raise ValueError("Block dimensions must be positive.")

    height, width, channels = image.shape

    pad_height = (block_height - height % block_height) % block_height
    pad_width = (block_width - width % block_width) % block_width

    padded_image = np.pad(
        image,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="edge"
    )

    return padded_image


def image_to_blocks(
    image: np.ndarray,
    block_size: tuple[int, int]
) -> np.ndarray:

    block_height, block_width = block_size

    height, width, channels = image.shape

    if height % block_height != 0 or width % block_width != 0:
        raise ValueError("Image dimensions must be divisible by block size.")

    blocks = image.reshape(
        height // block_height,
        block_height,
        width // block_width,
        block_width,
        channels
    )

    blocks = blocks.transpose(0, 2, 1, 3, 4)

    return blocks.reshape(
        -1,
        block_height,
        block_width,
        channels
    )


def blocks_to_vectors(blocks: np.ndarray) -> np.ndarray:

    return blocks.reshape(len(blocks), -1).astype(np.float32)


def prepare_image(image: np.ndarray, block_size: tuple[int, int]):

    padded_image = pad_image(image, block_size)

    blocks = image_to_blocks(padded_image, block_size)

    vectors = blocks_to_vectors(blocks)

    return padded_image, blocks, vectors