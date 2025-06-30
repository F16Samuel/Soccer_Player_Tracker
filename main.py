# main.py

"""
CLI entrypoint to run the football tracking pipeline.
"""

import argparse
from ml_app.classification import process_soccer_video


def main():
    parser = argparse.ArgumentParser(description="Football Player Tracking Pipeline")
    parser.add_argument(
        "--input", type=str, default="videos/input/15sec_input_720p.mp4",
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", type=str, default="videos/output/15sec_input_720p.mp4",
        help="Path to save output video file"
    )
    args = parser.parse_args()

    process_soccer_video(source_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()