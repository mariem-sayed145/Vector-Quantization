import json

import numpy as np
import pytest
from PIL import Image

from src.compression import encode_indices, save_binary_stream, save_codebook
from src.decompression import decompress_image, reconstruct_image


def test_decompress_round_trip_with_padding(tmp_path):
    """Decompression restores RGB dimensions and removes compression padding."""
    block_size = (2, 3)
    original_shape = (5, 7, 3)
    padded_shape = (6, 9, 3)

    # A small, known codebook: each vector represents one 2x3 RGB block.
    codebook = np.array(
        [
            [255, 0, 0] * 6,
            [0, 255, 0] * 6,
            [0, 0, 255] * 6,
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 1, 0, 2, 2, 1, 0], dtype=np.int32)

    codebook_path = tmp_path / "codebook.json"
    binary_path = tmp_path / "compressed.txt"
    output_path = tmp_path / "decompressed.png"

    save_codebook(
        codebook=codebook,
        original_shape=original_shape,
        padded_shape=padded_shape,
        block_size=block_size,
        output_path=codebook_path,
    )
    save_binary_stream(
        encode_indices(indices, k=len(codebook)),
        binary_path,
    )

    result = decompress_image(codebook_path, binary_path, output_path)

    assert output_path.exists()
    assert result["num_blocks"] == 9
    assert result["codebook_size"] == 3

    reconstructed = np.array(Image.open(output_path).convert("RGB"))
    assert reconstructed.shape == original_shape

    expected = reconstruct_image(
        indices=indices,
        codebook=codebook,
        original_shape=original_shape,
        padded_shape=padded_shape,
        block_size=block_size,
    )
    assert np.array_equal(reconstructed, expected)


def test_decompress_single_entry_codebook(tmp_path):
    codebook = np.array([[120, 80, 40] * 4], dtype=np.float32)
    codebook_path = tmp_path / "codebook.json"
    binary_path = tmp_path / "compressed.txt"
    output_path = tmp_path / "result.png"

    save_codebook(
        codebook,
        original_shape=(2, 4, 3),
        padded_shape=(2, 4, 3),
        block_size=(2, 2),
        output_path=codebook_path,
    )
    save_binary_stream(encode_indices(np.array([0, 0]), k=1), binary_path)

    decompress_image(codebook_path, binary_path, output_path)

    image = np.array(Image.open(output_path).convert("RGB"))
    assert image.shape == (2, 4, 3)
    assert np.all(image == np.array([120, 80, 40], dtype=np.uint8))


def test_decompress_rejects_wrong_binary_stream_length(tmp_path):
    codebook = np.array([[10, 20, 30] * 4, [40, 50, 60] * 4], dtype=np.float32)
    codebook_path = tmp_path / "codebook.json"
    binary_path = tmp_path / "compressed.txt"

    save_codebook(codebook, (2, 4, 3), (2, 4, 3), (2, 2), codebook_path)
    save_binary_stream("0", binary_path)  # Two indices require two bits.

    with pytest.raises(ValueError, match="binary stream length"):
        decompress_image(codebook_path, binary_path, tmp_path / "out.png")


def test_decompress_rejects_mismatched_codebook_vector_size(tmp_path):
    codebook_path = tmp_path / "codebook.json"
    binary_path = tmp_path / "compressed.txt"

    # Metadata describes a 2x2 RGB block (12 values), but this vector has 6.
    data = {
        "codebook": [[1, 2, 3, 4, 5, 6]],
        "original_shape": [2, 2, 3],
        "padded_shape": [2, 2, 3],
        "block_size": [2, 2],
        "codebook_size": 1,
        "bits_per_index": 1,
    }
    codebook_path.write_text(json.dumps(data), encoding="utf-8")
    binary_path.write_text("0", encoding="ascii")

    with pytest.raises(ValueError, match="Codebook vector size"):
        decompress_image(codebook_path, binary_path, tmp_path / "out.png")


def test_decompress_rejects_invalid_binary_characters(tmp_path):
    codebook = np.array([[10, 20, 30] * 4], dtype=np.float32)
    codebook_path = tmp_path / "codebook.json"
    binary_path = tmp_path / "compressed.txt"

    save_codebook(codebook, (2, 2, 3), (2, 2, 3), (2, 2), codebook_path)
    save_binary_stream("x", binary_path)

    with pytest.raises(ValueError, match="invalid characters"):
        decompress_image(codebook_path, binary_path, tmp_path / "out.png")


def test_decompression_uses_actual_compression_outputs(tmp_path):
    """The decompressor accepts the exact artifacts emitted by compress_image."""
    from src.compression import compress_image

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "compressed"
    input_dir.mkdir()

    source = np.random.default_rng(7).integers(
        0, 256, size=(7, 8, 3), dtype=np.uint8
    )
    source_path = input_dir / "source.png"
    Image.fromarray(source).save(source_path)

    compression_result = compress_image(
        image_path=source_path,
        block_size=(2, 3),
        codebook_size=4,
        output_dir=output_dir,
    )

    decompression_result = decompress_image(
        codebook_path=compression_result["codebook_path"],
        binary_path=compression_result["binary_path"],
        output_path=tmp_path / "decompressed.png",
    )

    output = np.array(Image.open(decompression_result["output_path"]).convert("RGB"))
    assert output.shape == source.shape
    assert decompression_result["num_blocks"] == compression_result["num_blocks"]
