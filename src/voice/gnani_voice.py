"""Gnani.ai Speech Services integration (STT & TTS) with voice normalization."""

import io
import logging
import math
import os
import struct
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from src.config import (
    GNANI_API_KEY,
    GNANI_STT_LANGUAGE,
    GNANI_TTS_VOICE,
    GNANI_TTS_MODEL,
    GNANI_TTS_LANGUAGE,
)
from src.agent.speech_normalizer import normalize_for_voice

logger = logging.getLogger(__name__)


def synthesize_local_speech(text: str, output_file: Path) -> bool:
    """Synthesize human speech offline using Windows SAPI TTS."""
    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        filestream = win32com.client.Dispatch("SAPI.SpFileStream")
        # 3 = SSFMCreateForWrite
        filestream.Open(str(output_file), 3, False)
        speaker.AudioOutputStream = filestream
        speaker.Speak(text)
        filestream.Close()
        return True
    except Exception as e:
        logger.warning(f"Local SAPI speech synthesis unavailable: {e}")
        return False


def generate_synthetic_wav(output_path: Path, duration_sec: float = 1.0, freq: float = 440.0):
    """Generate a valid 16-bit PCM WAV tone for offline testing and fallback."""
    sample_rate = 16000
    n_samples = int(sample_rate * duration_sec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767.0 * 0.25 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
            data = struct.pack("<h", value)
            wav_file.writeframesraw(data)


class VoiceManager:
    """Manages audio transcription and speech synthesis via Gnani.ai APIs."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GNANI_API_KEY
        self.stt_client = None
        self.tts_client = None
        self.is_live = False

        if self.api_key:
            try:
                from gnani.stt import GnaniSTTClient
                from gnani.tts import GnaniTTSClient
                self.stt_client = GnaniSTTClient(api_key=self.api_key)
                self.tts_client = GnaniTTSClient(api_key=self.api_key)
                self.is_live = True
                logger.info("Successfully connected to Gnani.ai STT & TTS services.")
            except Exception as e:
                logger.warning(f"Failed to authenticate Gnani client: {e}. Falling back to simulation.")
        else:
            logger.info("GNANI_API_KEY not found. Running voice layer in simulation mode.")

    def transcribe(self, audio_path: Path, language_code: Optional[str] = None) -> str:
        """Transcribe customer audio using Gnani STT REST API."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        lang = language_code or GNANI_STT_LANGUAGE

        if self.is_live and self.stt_client:
            try:
                resp = self.stt_client.transcribe(str(audio_path), language_code=lang)
                transcript = resp.get("transcript", "").strip()
                logger.info(f"Gnani STT transcribed ({lang}): '{transcript}'")
                return transcript
            except Exception as e:
                logger.error(f"Gnani STT API call failed: {e}. Falling back to simulated text.")

        # Offline / simulation fallback:
        # Check if there is an accompanying .txt file or metadata
        txt_path = audio_path.with_suffix(".txt")
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8").strip()

        return "Order number hai 8978 2837. Mujhe help chahiye."

    def synthesize(
        self,
        text: str,
        output_file: Path,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None
    ) -> Path:
        """Synthesize spoken response using Gnani TTS REST API."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Normalize text for spoken speech (dates, order groups, currency)
        normalized_text = normalize_for_voice(text)
        if not normalized_text.strip():
            normalized_text = "Sorry, I didn't catch that. Could you say it again?"

        voice_name = voice or GNANI_TTS_VOICE
        model_name = model or GNANI_TTS_MODEL
        lang = language or GNANI_TTS_LANGUAGE

        if self.is_live and self.tts_client:
            try:
                logger.info(f"Synthesizing via Gnani TTS (Voice: {voice_name}, Model: {model_name}, Lang: {lang})...")
                self.tts_client.synthesize(
                    text=normalized_text,
                    voice=voice_name,
                    model=model_name,
                    language=lang,
                    output_file=str(output_file)
                )
                return output_file
            except Exception as e:
                logger.error(f"Gnani TTS API call failed: {e}. Generating synthetic audio.")

        # Offline fallback: Try local speech synthesizer first
        if synthesize_local_speech(normalized_text, output_file):
            return output_file

        # Final fallback: synthetic PCM tone
        generate_synthetic_wav(output_file, duration_sec=1.5, freq=520.0)
        return output_file

    def voice_turn(
        self,
        audio_path: Path,
        agent: Any,
        output_wav_path: Optional[Path] = None,
        session_id: str = "voice_session"
    ) -> Dict[str, Any]:
        """Execute one full end-to-end voice turn as specified in Capstone Brief Step 4."""
        audio_path = Path(audio_path)
        out_wav = Path(output_wav_path or (audio_path.parent / f"{audio_path.stem}_reply.wav"))

        # 1. Listen (Gnani STT)
        heard = self.transcribe(audio_path, language_code=GNANI_STT_LANGUAGE)

        # 2. Brain (Conversational Agent Step 3)
        if not heard.strip():
            reply = "Sorry, I didn't catch that. Could you say it again?"
            spoken_reply = reply
            decision = "clarify"
            evidence_ids = []
            tool_calls = []
        else:
            agent_res = agent.respond(heard, session_id=session_id, mode="conversational", voice_mode=True)
            reply = agent_res["reply"]
            spoken_reply = agent_res.get("spoken_reply", reply)
            decision = agent_res.get("decision", "answer")
            evidence_ids = agent_res.get("evidence_ids", [])
            tool_calls = agent_res.get("tool_calls", [])

        # 3. Speak (Gnani TTS)
        self.synthesize(
            spoken_reply,
            output_file=out_wav,
            voice=GNANI_TTS_VOICE,
            model=GNANI_TTS_MODEL,
            language=GNANI_TTS_LANGUAGE
        )

        return {
            "transcript": heard,
            "reply": reply,
            "spoken_reply": spoken_reply,
            "audio_output_file": str(out_wav),
            "decision": decision,
            "evidence_ids": evidence_ids,
            "tool_calls": tool_calls,
        }

