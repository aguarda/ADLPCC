import argparse
import sys
import os
import numpy as np

from absl import app
from absl.flags import argparse_flags

import pc2vox


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "input_file",
        help="Input Point Cloud filename (.ply).")
    parser.add_argument(
        "bitdepth", type=int,
        help="Desired Point Cloud bit depth precision.")
    # Parse arguments
    args = parser.parse_args(argv[1:])
    if args.input_file is None or args.bitdepth is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    in_file = args.input_file
    if not in_file.endswith('.ply'):
        raise ValueError("Input must be a PLY file (.ply extension).")

    out_file = os.path.splitext(in_file)[0] + "_vox" + str(args.bitdepth) + ".ply"
    # Load Point Cloud
    in_points = pc2vox.load_pc(in_file)
    # Determine original bit depth precision
    in_bitdepth = np.ceil(np.log2(in_points.max() + 1))
    # Compute the sampling scale based on input and target precisions
    scale = pow(2, args.bitdepth - in_bitdepth)
    # Apply the previous scale to the point coordinates
    out_points = np.floor((in_points * scale) + 0.5).astype(np.uint32)
    # Truncate values
    max_value = pow(2, args.bitdepth) - 1
    out_points[np.where(out_points > max_value)] = max_value
    # Remove duplicates
    out_points = np.unique(out_points, axis=0)
    # Save Point Cloud
    pc2vox.save_pc(out_points, out_file)

if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)