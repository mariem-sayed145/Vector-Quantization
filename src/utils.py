from pathlib import Path


def calculate_compression_ratio(
    original_size: int,
    compressed_size: int
) -> float:

    if compressed_size <= 0:
        raise ValueError("Compressed size must be positive.")

    return original_size / compressed_size


def calculate_space_saving(
    original_size: int,
    compressed_size: int
) -> float:

    if original_size <= 0:
        raise ValueError("Original size must be positive.")

    return (
        (original_size - compressed_size)
        / original_size
    ) * 100


def get_file_size(file_path: str | Path) -> int:

    return Path(file_path).stat().st_size