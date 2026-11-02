"""Configuration settings and environment variable management for VoiceOps."""

import os
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directory resolution
def find_dataset_dir() -> Path:
    env_path = os.getenv("DATASET_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path).resolve()

    candidates = [
        BASE_DIR / "Capstone_Project_Dataset_FTTT_2026",
        BASE_DIR / "dataset",
        BASE_DIR,
    ]
    for candidate in candidates:
        if (candidate / "data" / "orders.csv").exists() and (candidate / "kb").exists():
            return candidate.resolve()

    return (BASE_DIR / "Capstone_Project_Dataset_FTTT_2026").resolve()

DATASET_DIR = find_dataset_dir()
DATA_DIR = DATASET_DIR / "data"
KB_DIR = DATASET_DIR / "kb"
PRACTICE_DIR = DATASET_DIR / "practice"
TOOLS_DIR = DATASET_DIR / "tools"

# Frozen Reference Date for ZentroMart
# As per KB-00.2 and project brief: 20 October 2026
REFERENCE_DATE = date(2026, 10, 20)

# External API Credentials
GNANI_API_KEY = os.getenv("GNANI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Voice / Speech Settings (Gnani.ai)
GNANI_STT_LANGUAGE = os.getenv("GNANI_STT_LANGUAGE", "en-IN")
GNANI_TTS_VOICE = os.getenv("GNANI_TTS_VOICE", "Kaveri")
GNANI_TTS_MODEL = os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")
GNANI_TTS_LANGUAGE = os.getenv("GNANI_TTS_LANGUAGE", "en-IN")

# Escalation & Policy Rules
HIGH_VALUE_DAMAGE_THRESHOLD_INR = 10000
REPEAT_CONTACT_THRESHOLD = 2  # ESC-1: 3rd or later contact within 7 days -> contacts_last_7_days >= 2
OVERDUE_REFUND_BUFFER_DAYS = 3  # ESC-1: Refund more than 3 business days past max timeline
MAX_VERIFICATION_RETRIES = 1  # IDV-3: Allow 1 retry before denying order details

# LLM Model default
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

