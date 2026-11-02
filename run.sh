#!/usr/bin/env bash
# VoiceOps Batch Audio Runner Script
# Usage: ./run.sh [input_audio_dir] [output_dir]

set -e

INPUT_DIR="${1:-sample_outputs/input_audio}"
OUTPUT_DIR="${2:-sample_outputs/batch_run}"

echo "Starting VoiceOps Batch Audio Evaluation..."
echo "Input Directory:  $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"

python run.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"

echo "VoiceOps Batch Run Completed successfully."

