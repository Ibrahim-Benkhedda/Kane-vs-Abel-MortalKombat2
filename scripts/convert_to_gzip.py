import gzip
import sys
import os

def compress_state(input_file, output_file=None):
    """
    Compress a raw state file using gzip.

    Parameters:
        input_file: Path to the input (raw) state file.
        output_file: Optional path for the gzip-compressed output file.
                     If not provided, it will default to input_file + ".gz"
    """
    # Decide on a default output file name if none is provided
    if output_file is None:
        output_file = input_file + ".gz"

    # Read the contents of the input file and write them to a gzip-compressed file
    with open(input_file, "rb") as f_in:
        with gzip.open(output_file, "wb") as f_out:
            f_out.write(f_in.read())

    print(f"Compressed '{input_file}' to '{output_file}'")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python compress_state.py <input_file> [<output_file>]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) == 3 else None

    compress_state(input_path, output_path)