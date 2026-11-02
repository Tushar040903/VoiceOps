"""Official Batch Audio Execution Runner for VoiceOps.

Accepts a folder of customer audio files, runs the conversational agent on each,
and saves replies (both as text and audio) along with retrieved document chunks.
Organizers use this script to test all submissions uniformly.

Usage:
    python run.py --input_dir <path_to_audio_folder> --output_dir <path_to_output_folder>
    python run.py --practice  # Run on practice benchmark suite
"""

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Ensure UTF-8 output encoding across Windows PowerShell and Unix shells
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.data.loader import DataLoader
from src.agent.conversational_agent import ConversationalAgent
from src.voice.gnani_voice import VoiceManager, generate_synthetic_wav

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoiceOps-Runner")


def process_audio_batch(input_dir: Path, output_dir: Path, language: str = "en-IN") -> List[Dict[str, Any]]:
    """Process all .wav files in input_dir and output speech replies, transcripts, and citations."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader()
    agent = ConversationalAgent(data_loader=loader)
    voice = VoiceManager()

    audio_files = sorted(list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3")))

    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}. Generating a sample audio test file.")
        sample_file = input_dir / "sample_customer_query.wav"
        generate_synthetic_wav(sample_file, duration_sec=1.5)
        # Create accompanying transcript hint for simulation
        sample_file.with_suffix(".txt").write_text("Order number hai 8978 2837. Mera TV damaged aaya hai.", encoding="utf-8")
        audio_files = [sample_file]

    results: List[Dict[str, Any]] = []

    print(f"\n[VoiceOps] Processing {len(audio_files)} customer audio recording(s)...")
    print("-" * 65)

    for i, audio_path in enumerate(audio_files, 1):
        start_t = time.time()
        session_id = f"batch_session_{audio_path.stem}"
        out_wav_path = output_dir / f"{audio_path.stem}_reply.wav"

        logger.info(f"[{i}/{len(audio_files)}] Processing: {audio_path.name}")

        # Execute full voice turn: STT -> Agent Brain (RAG + Tools) -> TTS
        turn_result = voice.voice_turn(
            audio_path=audio_path,
            agent=agent,
            output_wav_path=out_wav_path,
            session_id=session_id
        )

        latency_sec = round(time.time() - start_t, 3)

        record = {
            "index": i,
            "input_file": str(audio_path),
            "transcript": turn_result["transcript"],
            "agent_reply": turn_result["reply"],
            "spoken_reply": turn_result["spoken_reply"],
            "output_audio_file": str(out_wav_path),
            "retrieved_section_ids": turn_result["evidence_ids"],
            "evidence_citations": turn_result["evidence_ids"],
            "decision": turn_result["decision"],
            "tool_calls": turn_result["tool_calls"],
            "latency_seconds": latency_sec,
        }
        results.append(record)

        print(f"-> Heard:      \"{turn_result['transcript']}\"")
        print(f"-> Decision:   {turn_result['decision']}")
        print(f"-> Chunks:     {turn_result['evidence_ids']}")
        print(f"-> Tools:      {turn_result['tool_calls']}")
        print(f"-> Reply:      \"{turn_result['spoken_reply'][:90]}...\"")
        print(f"-> Audio saved:{out_wav_path.name} ({latency_sec}s)\n")

    summary_file = output_dir / "results.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"total_processed": len(results), "records": results}, f, indent=2)

    logger.info(f"Batch run completed! Results saved to {summary_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="VoiceOps 2026: Official Batch Execution Runner")
    parser.add_argument("--input_dir", "--audio_dir", type=str, default="sample_outputs/input_audio",
                        help="Folder containing input customer audio files (.wav)")
    parser.add_argument("--output_dir", type=str, default="sample_outputs/batch_run",
                        help="Folder to save text replies, audio responses, and retrieved citations")
    parser.add_argument("--language", type=str, default="en-IN",
                        help="STT language code (default: en-IN)")
    parser.add_argument("--practice", action="store_true",
                        help="Run full capstone benchmark evaluation on practice questions & conversations")

    args = parser.parse_args()

    if args.practice:
        from eval_pipeline import run_full_evaluation
        run_full_evaluation()
        return

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    process_audio_batch(in_dir, out_dir, language=args.language)


if __name__ == "__main__":
    main()
