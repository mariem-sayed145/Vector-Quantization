import numpy as np

from src.utils import calculate_compression_ratio, calculate_space_saving


def packed_compressed_bits(
    num_blocks: int,
    codebook_size: int,
    block_size: tuple[int, int],
    channels: int = 3,
) -> dict:
    """
    Ideal compressed size if the stream were bit-packed:
    indices + codebook (8 bits/value) + metadata (6 x 32-bit ints).
    """
    bits_per_index = max(1, (codebook_size - 1).bit_length())
    index_bits = num_blocks * bits_per_index
    codebook_bits = codebook_size * block_size[0] * block_size[1] * channels * 8
    metadata_bits = 6 * 32
    return {
        "index_bits": index_bits,
        "codebook_bits": codebook_bits,
        "metadata_bits": metadata_bits,
        "total_bytes": (index_bits + codebook_bits + metadata_bits + 7) // 8,
    }


def mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.mean(diff ** 2))


def psnr(mse: float) -> float:
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def compression_report(result: dict, block_size: tuple[int, int]) -> dict:
    """Both ratios: actual files on disk, and ideal bit-packed."""
    packed = packed_compressed_bits(
        result["num_blocks"], result["codebook_size"], block_size
    )
    original = result["original_size"]
    return {
        "file_ratio": calculate_compression_ratio(original, result["compressed_size"]),
        "file_saving": calculate_space_saving(original, result["compressed_size"]),
        "packed_bytes": packed["total_bytes"],
        "packed_ratio": calculate_compression_ratio(original, packed["total_bytes"]),
        "packed_saving": calculate_space_saving(original, packed["total_bytes"]),
    }