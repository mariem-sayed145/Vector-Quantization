from pathlib import Path

from src.compression import compress_image
from src.decompression import decompress_image
from src.image_processing import load_image
from src.metrics import compression_report, mean_squared_error, psnr


def ask_path(prompt: str, default: str) -> str:
    return input(f"{prompt} [{default}]: ").strip() or default


def run_compression():
    image_name = input("\nEnter image name (e.g. test.jpg): ").strip()
    image_path = Path("data/input") / image_name

    block_height = int(input("Enter block height: "))
    block_width = int(input("Enter block width: "))
    codebook_size = int(input("Enter codebook size: "))
    block_size = (block_height, block_width)

    result = compress_image(
        image_path=image_path,
        block_size=block_size,
        codebook_size=codebook_size,
    )

    original = load_image(image_path)
    reconstructed = load_image(result["reconstructed_path"])
    mse = mean_squared_error(original, reconstructed)
    report = compression_report(result, block_size)

    print("\n" + "=" * 50)
    print("             COMPRESSION RESULTS")
    print("=" * 50)
    print(f"Original Size:          {result['original_size']:,} bytes")
    print(f"Codebook File Size:     {result['codebook_file_size']:,} bytes")
    print(f"Binary Stream Size:     {result['binary_file_size']:,} bytes")
    print(f"Compressed Size (files):{result['compressed_size']:,} bytes")
    print(f"Compression Ratio (files):      {report['file_ratio']:.2f}:1 "
          f"({report['file_saving']:.1f}% saved)")
    print(f"Compression Ratio (bit-packed): {report['packed_ratio']:.2f}:1 "
          f"({report['packed_bytes']:,} bytes, {report['packed_saving']:.1f}% saved)")
    print(f"Number of Blocks:       {result['num_blocks']}")
    print(f"Bits Per Index:         {result['bits_per_index']}")
    print(f"MSE:                    {mse:.4f}")
    print(f"PSNR:                   {psnr(mse):.2f} dB")
    print("\nGenerated Files:")
    print(f"  Codebook:          {result['codebook_path']}")
    print(f"  Compressed Stream: {result['binary_path']}")
    print(f"  Reconstructed:     {result['reconstructed_path']}")


def run_decompression():
    result = decompress_image(
        codebook_path=ask_path("\nCodebook file", "data/output/codebook.json"),
        binary_path=ask_path("Compressed stream file", "data/output/compressed.txt"),
        output_path=ask_path("Output image", "data/output/decompressed.png"),
    )
    height, width = result["image_shape"][:2]

    print("\nDecompression completed successfully!")
    print(f"Output image:     {result['output_path']}")
    print(f"Dimensions:       {width} x {height}")
    print(f"Number of blocks: {result['num_blocks']}")
    print(f"Codebook size:    {result['codebook_size']}")
    print(f"Bits per index:   {result['bits_per_index']}")


def main():
    actions = {"1": run_compression, "2": run_decompression}

    while True:
        print("\n" + "=" * 50)
        print("       VECTOR QUANTIZATION")
        print("=" * 50)
        print("1. Compress an image")
        print("2. Decompress an image")
        print("3. Exit")

        choice = input("Choose an option: ").strip()

        if choice == "3":
            break
        if choice not in actions:
            print("Invalid option.")
            continue

        try:
            actions[choice]()
        except (ValueError, FileNotFoundError, OSError, KeyError) as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()