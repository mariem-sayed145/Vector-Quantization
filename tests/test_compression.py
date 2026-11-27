
import numpy as np
import pytest


from src.compression import (
    encode_indices,
    decode_indices,
    save_codebook,
    load_codebook,
    save_binary_stream
)


def test_binary_encoding():

    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    binary_stream = encode_indices(indices, k=8)

    decoded = decode_indices(
        binary_stream,
        num_indices=len(indices),
        k=8
    )

    assert np.array_equal(indices, decoded)


def test_non_power_of_two_codebook():

    indices = np.array([0, 1, 2, 3, 4])

    binary_stream = encode_indices(indices, k=5)

    decoded = decode_indices(
        binary_stream,
        num_indices=len(indices),
        k=5
    )

    assert np.array_equal(indices, decoded)


def test_single_vector_codebook():

    indices = np.array([0, 0, 0, 0])

    binary_stream = encode_indices(indices, k=1)

    decoded = decode_indices(
        binary_stream,
        num_indices=4,
        k=1
    )

    assert np.array_equal(indices, decoded)


def test_invalid_index():

    indices = np.array([0, 1, 5])

    with pytest.raises(ValueError):
        encode_indices(indices, k=4)


def test_invalid_binary_stream():

    with pytest.raises(ValueError):
        decode_indices(
            "0123",
            num_indices=2,
            k=4
        )


def test_codebook_save_and_load(tmp_path):

    codebook = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=np.float32)

    path = tmp_path / "codebook.json"

    save_codebook(
        codebook,
        (10, 10, 3),
        (10, 10, 3),
        (2, 2),
        path
    )

    loaded = load_codebook(path)

    assert np.allclose(
        loaded["codebook"],
        codebook
    )

    assert loaded["original_shape"] == (10, 10, 3)


def test_save_binary_stream(tmp_path):

    binary_stream = "01010101"

    path = tmp_path / "compressed.txt"

    save_binary_stream(binary_stream, path)

    assert path.exists()

    assert path.read_text() == binary_stream


