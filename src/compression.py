
import json
import numpy as np

from pathlib import Path


from src.image_processing import load_image, prepare_image
from src.codebook import generate_codebook, find_nearest_centroids


def encode_indices(indices: np.ndarray, k: int) -> str:
    """
    Encode codebook indices into a binary string.
    """

    if k < 1:
        raise ValueError("Codebook size must be positive.")

    if np.any(indices < 0) or np.any(indices >= k):
        raise ValueError("Invalid codebook indices.")

    bits_per_index = max(1, (k - 1).bit_length())

    return "".join(
        format(int(index), f"0{bits_per_index}b")
        for index in indices
    )


def decode_indices(
    binary_stream: str,
    num_indices: int,
    k: int
) -> np.ndarray:
    """
    Decode binary string back into codebook indices.
    """

    if k < 1:
        raise ValueError("Codebook size must be positive.")

    bits_per_index = max(1, (k - 1).bit_length())
    required_bits = num_indices * bits_per_index

    if len(binary_stream) != required_bits:
        raise ValueError("Invalid binary stream length.")

    if any(bit not in "01" for bit in binary_stream):
        raise ValueError("Binary stream contains invalid characters.")

    indices = []

    for i in range(0, required_bits, bits_per_index):
        chunk = binary_stream[i:i + bits_per_index]
        index = int(chunk, 2)

        if index >= k:
            raise ValueError("Invalid codebook index.")

        indices.append(index)

    return np.array(indices, dtype=np.int32)


def save_codebook(
    codebook: np.ndarray,
    original_shape: tuple,
    padded_shape: tuple,
    block_size: tuple,
    output_path: str | Path
):
    """
    Save codebook and image metadata as a JSON file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "codebook": codebook.tolist(),
        "original_shape": list(original_shape),
        "padded_shape": list(padded_shape),
        "block_size": list(block_size),
        "codebook_size": len(codebook),
        "bits_per_index": max(1, (len(codebook) - 1).bit_length())
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_codebook(codebook_path: str | Path) -> dict:
    """
    Load codebook and metadata from JSON.
    """

    with open(codebook_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    data["codebook"] = np.array(
        data["codebook"],
        dtype=np.float32
    )

    data["original_shape"] = tuple(data["original_shape"])
    data["padded_shape"] = tuple(data["padded_shape"])
    data["block_size"] = tuple(data["block_size"])

    return data


def save_binary_stream(
    binary_stream: str,
    output_path: str | Path
):
    """
    Save the binary stream as a readable text file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(binary_stream)


def compress_image(
    image_path: str | Path,
    block_size: tuple[int, int],
    codebook_size: int,
    output_dir: str | Path = "data/output"
) -> dict:
    """
    Compress an RGB image using Vector Quantization.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load RGB image
    image = load_image(image_path)

    # 2. Prepare image
    padded_image, blocks, vectors = prepare_image(
        image,
        block_size
    )

    # 3. Generate codebook
    codebook = generate_codebook(
        vectors,
        codebook_size
    )

    # 4. Find nearest codebook vector
    indices, distances = find_nearest_centroids(
        vectors,
        codebook
    )

    # 5. Encode indices
    binary_stream = encode_indices(
        indices,
        codebook_size
    )

    # 6. Define output paths
    codebook_path = output_dir / "codebook.json"
    binary_path = output_dir / "compressed.txt"
    reconstructed_path = output_dir / "reconstructed.png"

    # 7. Save codebook and metadata
    save_codebook(
        codebook,
        image.shape,
        padded_image.shape,
        block_size,
        codebook_path
    )

    # 8. Save compressed stream
    save_binary_stream(
        binary_stream,
        binary_path
    )

    # 10. Calculate statistics
    original_size = image.nbytes

    codebook_file_size = codebook_path.stat().st_size
    stream_file_size = binary_path.stat().st_size

    compressed_size = (
        codebook_file_size + stream_file_size
    )

    compression_ratio = (
        original_size / compressed_size
        if compressed_size > 0
        else 0
    )

    bits_per_index = max(
        1,
        (codebook_size - 1).bit_length()
    )

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "codebook_file_size": codebook_file_size,
        "binary_file_size": stream_file_size,
        "compression_ratio": compression_ratio,
        "codebook_size": codebook_size,
        "bits_per_index": bits_per_index,
        "num_blocks": len(blocks),
        "codebook_path": str(codebook_path),
        "binary_path": str(binary_path),
        "mean_squared_error": float(np.mean(distances))
    }
