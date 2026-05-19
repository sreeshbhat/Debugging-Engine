import json
import os
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
STORAGE_DIR = ROOT_DIR / "storage"
CHALLENGES_PATH = DATA_DIR / "challenges.json"
RESULTS_PATH = STORAGE_DIR / "results.json"


def get_secret(key: str, default: Any = None) -> Any:
    """Read configuration from Streamlit secrets first, then environment."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


def ensure_json_file(path: Path, default_value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(default_value, indent=2), encoding="utf-8")


def load_challenges(path: Path = CHALLENGES_PATH) -> list[dict]:
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def parse_json_response(raw_text: str) -> dict:
    """Try to recover a JSON object from a model response."""
    if not raw_text:
        raise ValueError("Empty response from model.")

    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model response did not contain valid JSON.") from None
        return json.loads(raw_text[start : end + 1])


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def grade_from_score(score: int) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Average"
    return "Poor"


def badge_from_score(score: int) -> str:
    if score >= 90:
        return "Debugging Prompt Master"
    if score >= 80:
        return "AI Coding Pro"
    if score >= 60:
        return "Good Debugger"
    if score >= 40:
        return "Needs Better Context"
    return "Prompt Too Vague"


def deterministic_provider_bucket(student_roll: str, challenge_id: str, providers: list[str]) -> str:
    if not providers:
        raise ValueError("No configured providers are available for load splitting.")

    bucket_key = f"{student_roll.strip().lower()}::{challenge_id.strip().lower()}"
    digest = sha256(bucket_key.encode("utf-8")).hexdigest()
    index = int(digest, 16) % len(providers)
    return providers[index]


def filter_student_submissions(submissions: list[dict], student_roll: str) -> list[dict]:
    normalized_roll = student_roll.strip().lower()
    return [
        item
        for item in submissions
        if str(item.get("student_roll", "")).strip().lower() == normalized_roll
    ]


def format_attempt_time(value: str) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return value


def leaderboard_dataframe(submissions: list[dict]) -> pd.DataFrame:
    if not submissions:
        return pd.DataFrame(
            columns=["student_name", "student_roll", "challenge_title", "score", "grade", "badge", "attempt_time"]
        )

    rows = []
    for item in submissions:
        rows.append(
            {
                "student_name": item.get("student_name", "Unknown"),
                "student_roll": item.get("student_roll", ""),
                "challenge_title": item.get("challenge_title", "Unknown"),
                "score": item.get("score", 0),
                "grade": item.get("grade", "Poor"),
                "badge": item.get("badge", badge_from_score(item.get("score", 0))),
                "attempt_time": format_attempt_time(item.get("created_at", "")),
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(by=["score", "attempt_time"], ascending=[False, False])
    return frame.reset_index(drop=True)


def submissions_to_csv_bytes(submissions: list[dict]) -> bytes:
    frame = pd.DataFrame(submissions)
    return frame.to_csv(index=False).encode("utf-8")
