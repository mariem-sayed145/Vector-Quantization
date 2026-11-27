from pathlib import Path

from src.compression import compress_image
from src.decompression import decompress_image
from src.metrics import psnr


def ask_path(prompt: str, default: str) -> str:
    return input(f"{prompt} [{default}]: ").strip() or default


def print_separator():
    print("=" * 65)


def run_compression():

    print_separator()
    print("             VECTOR QUANTIZATION COMPRESSION")
    print_separator()

    image_name = input(
        "\nEnter image name (e.g. test.jpg): "
    ).strip()

    image_path = Path("data/input") / image_name

    block_height = int(input("Enter block height: "))
    block_width = int(input("Enter block width: "))
    codebook_size = int(input("Enter codebook size: "))

    block_size = (block_height, block_width)

    result = compress_image(
        image_path=image_path,
        block_size=block_size,
        codebook_size=codebook_size
    )

    height, width, channels = result["original_shape"]

    padded_height, padded_width, _ = result["padded_shape"]

    print("\n")
    print_separator()
    print("                 STEP-BY-STEP CALCULATIONS")
    print_separator()

    # STEP 1: Original image dimensions.
    print("\n[STEP 1] Original Image")

    print(f"Height   = {height}")
    print(f"Width    = {width}")
    print(f"Channels = {channels}")

    print(
        f"Original image shape = "
        f"({height}, {width}, {channels})"
    )

    # STEP 2: Padding calculation.
    print("\n[STEP 2] Padding Calculation")

    padding_height = result["padding_height"]
    padding_width = result["padding_width"]

    print(
        f"Padding Height = "
        f"({block_height} - ({height} % {block_height})) "
        f"% {block_height}"
    )

    print(f"Padding Height = {padding_height}")

    print(
        f"Padding Width = "
        f"({block_width} - ({width} % {block_width})) "
        f"% {block_width}"
    )

    print(f"Padding Width = {padding_width}")

    print(
        f"Padded Height = {height} + {padding_height} "
        f"= {padded_height}"
    )

    print(
        f"Padded Width = {width} + {padding_width} "
        f"= {padded_width}"
    )

    # STEP 3: Number of blocks.
    print("\n[STEP 3] Number of Blocks")

    blocks_per_row = padded_width // block_width
    block_rows = padded_height // block_height

    print(
        f"Blocks per row = {padded_width} / {block_width} "
        f"= {blocks_per_row}"
    )

    print(
        f"Block rows = {padded_height} / {block_height} "
        f"= {block_rows}"
    )

    print(
        f"Total Blocks = {block_rows} * {blocks_per_row} "
        f"= {result['num_blocks']}"
    )

    # STEP 4: Vector dimensions.
    print("\n[STEP 4] Vector Dimensions")

    vector_length = (
        block_height * block_width * channels
    )

    print(
        f"Vector Length = Block Height * Block Width * Channels"
    )

    print(
        f"Vector Length = {block_height} * "
        f"{block_width} * {channels}"
    )

    print(f"Vector Length = {vector_length}")

    # STEP 5: Bits per index.
    print("\n[STEP 5] Bits Required for Each Index")

    print("Formula: ceil(log2(K))")

    print(f"K = {codebook_size}")

    print(
        f"Bits per Index = ceil(log2({codebook_size})) "
        f"= {result['bits_per_index']}"
    )

    # STEP 6: Total index bits.
    print("\n[STEP 6] Total Bits for Encoded Indices")

    print(
        f"Total Index Bits = Number of Blocks * Bits per Index"
    )

    print(
        f"Total Index Bits = {result['num_blocks']} * "
        f"{result['bits_per_index']}"
    )

    print(f"Total Index Bits = {result['index_bits']} bits")

    print(
        f"Binary Stream Length = "
        f"{result['binary_stream_length']} bits"
    )

    # STEP 7: Codebook calculation.
    print("\n[STEP 7] Codebook Calculation")

    print(f"Number of Codewords = {codebook_size}")

    print(f"Values in each Codeword = {vector_length}")

    print(
        f"Total Codebook Values = "
        f"{codebook_size} * {vector_length}"
    )

    print(
        f"Total Codebook Values = "
        f"{result['codebook_values']}"
    )

    print(
        f"Raw Codebook Size = "
        f"{result['codebook_values']} * 4 bytes"
    )

    print(
        f"Raw Codebook Size = "
        f"{result['codebook_raw_bytes']:,} bytes"
    )

    # STEP 8: Original size.
    print("\n[STEP 8] Original Image Size")

    print("Formula: Height * Width * Channels * 8 bits")

    print(
        f"Original Size = {height} * {width} * "
        f"{channels} * 1 byte"
    )

    print(
        f"Original Size = {result['original_size']:,} bytes"
    )

    # STEP 9: Compressed size.
    print("\n[STEP 9] Compressed Size")

    print(
        f"Codebook File = "
        f"{result['codebook_file_size']:,} bytes"
    )

    print(
        f"Binary Stream File = "
        f"{result['binary_file_size']:,} bytes"
    )

    print("Compressed Size = Codebook File + Binary Stream File")

    print(
        f"Compressed Size = "
        f"{result['codebook_file_size']} + "
        f"{result['binary_file_size']}"
    )

    print(
        f"Compressed Size = "
        f"{result['compressed_size']:,} bytes"
    )

    # STEP 10: Compression ratio.
    print("\n[STEP 10] Compression Ratio")

    print("Formula: Original Size / Compressed Size")

    print(
        f"Compression Ratio = "
        f"{result['original_size']} / "
        f"{result['compressed_size']}"
    )

    print(
        f"Compression Ratio = "
        f"{result['compression_ratio']:.4f}:1"
    )

    print("\nSpace Saving Calculation:")

    print(
        "Space Saving = "
        "(Original Size - Compressed Size) / Original Size * 100"
    )

    print(
        f"Space Saving = "
        f"({result['original_size']} - "
        f"{result['compressed_size']}) / "
        f"{result['original_size']} * 100"
    )

    print(
        f"Space Saving = "
        f"{result['space_saving']:.2f}%"
    )

    # STEP 11: Image quality.
    print("\n[STEP 11] Reconstruction Quality")

    mse = result["mean_squared_error"]

    print("MSE = Sum of Squared Pixel Differences / Total Pixels")

    print(f"MSE = {mse:.6f}")

    print(f"PSNR = 10 * log10(255^2 / MSE)")

    print(f"PSNR = {psnr(mse):.4f} dB")

    # Final result.
    print("\n")
    print_separator()
    print("                 FINAL RESULTS")
    print_separator()

    print(f"Original Size:       {result['original_size']:,} bytes")
    print(f"Compressed Size:     {result['compressed_size']:,} bytes")
    print(f"Compression Ratio:   {result['compression_ratio']:.4f}:1")
    print(f"Space Saving:        {result['space_saving']:.2f}%")
    print(f"Number of Blocks:    {result['num_blocks']}")
    print(f"Codebook Size:       {codebook_size}")
    print(f"Bits per Index:      {result['bits_per_index']}")
    print(f"MSE:                 {mse:.6f}")
    print(f"PSNR:                {psnr(mse):.4f} dB")

    print("\nGenerated Files:")

    print(f"Codebook:          {result['codebook_path']}")
    print(f"Compressed Stream: {result['binary_path']}")
    print(f"Reconstructed:     {result['reconstructed_path']}")

    print_separator()


def run_decompression():

    print_separator()
    print("             VECTOR QUANTIZATION DECOMPRESSION")
    print_separator()

    result = decompress_image(
        codebook_path=ask_path(
            "Codebook file",
            "data/output/codebook.json"
        ),
        binary_path=ask_path(
            "Compressed stream file",
            "data/output/compressed.txt"
        ),
        output_path=ask_path(
            "Output image",
            "data/output/decompressed.png"
        )
    )

    height, width = result["image_shape"][:2]

    print("\nDecompression completed successfully!")

    print(f"Output Image:     {result['output_path']}")
    print(f"Dimensions:       {width} x {height}")
    print(f"Number of Blocks: {result['num_blocks']}")
    print(f"Codebook Size:    {result['codebook_size']}")
    print(f"Bits per Index:   {result['bits_per_index']}")


def main():

    actions = {
        "1": run_compression,
        "2": run_decompression
    }

    while True:

        print("\n")
        print_separator()
        print("                 VECTOR QUANTIZATION")
        print_separator()

        print("1. Compress an image")
        print("2. Decompress an image")
        print("3. Exit")

        choice = input("\nChoose an option: ").strip()

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