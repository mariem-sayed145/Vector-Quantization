
from pathlib import Path

from src.compression import compress_image


def main():

    print("=" * 50)
    print("       VECTOR QUANTIZATION COMPRESSION")
    print("=" * 50)

    image_name = input(
        "\nEnter image name (e.g. test.jpg): "
    ).strip()

    image_path = Path("data/input") / image_name

    try:

        block_height = int(input("Enter block height: "))
        block_width = int(input("Enter block width: "))
        codebook_size = int(input("Enter codebook size: "))

        result = compress_image(
            image_path=image_path,
            block_size=(block_height, block_width),
            codebook_size=codebook_size
        )

        print("\nCompression completed successfully!")

        print("\n" + "=" * 50)
        print("             COMPRESSION RESULTS")
        print("=" * 50)

        print(
            f"\nOriginal Size: "
            f"{result['original_size']:,} bytes"
        )

        print(
            f"Compressed Size: "
            f"{result['compressed_size']:,} bytes"
        )

        print(
            f"Codebook File Size: "
            f"{result['codebook_file_size']:,} bytes"
        )

        print(
            f"Compressed Stream File Size: "
            f"{result['binary_file_size']:,} bytes"
        )

        print(
            f"\nCompression Ratio: "
            f"{result['compression_ratio']:.2f}:1"
        )

        print(
            f"Codebook Size: "
            f"{result['codebook_size']}"
        )

        print(
            f"Bits Per Index: "
            f"{result['bits_per_index']}"
        )

        print(
            f"Number of Blocks: "
            f"{result['num_blocks']}"
        )

        print(
            f"MSE: "
            f"{result['mean_squared_error']:.4f}"
        )

        print("\nGenerated Files:")

        print(f"Codebook: {result['codebook_path']}")
        print(f"Compressed Stream: {result['binary_path']}")
        print(f"Reconstructed Image: {result['reconstructed_path']}")

        print("\n" + "=" * 50)

    except (ValueError, FileNotFoundError, OSError) as error:

        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
