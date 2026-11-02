@echo off
REM VoiceOps Batch Audio Runner Script for Windows
REM Usage: run.bat [input_audio_dir] [output_dir]

set INPUT_DIR=%1
if "%INPUT_DIR%"=="" set INPUT_DIR=sample_outputs\input_audio

set OUTPUT_DIR=%2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=sample_outputs\batch_run

echo Starting VoiceOps Batch Audio Evaluation...
echo Input Directory:  %INPUT_DIR%
echo Output Directory: %OUTPUT_DIR%

python run.py --input_dir "%INPUT_DIR%" --output_dir "%OUTPUT_DIR%"

echo VoiceOps Batch Run Completed successfully.

