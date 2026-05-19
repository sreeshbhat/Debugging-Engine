import json
from pathlib import Path
from typing import Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from utils.helpers import ensure_json_file


class SubmissionRepository:
    def __init__(self, mongodb_uri: str | None, storage_path: Path):
        self.mongodb_uri = mongodb_uri
        self.storage_path = storage_path
        self.client = None
        self.collection = None
        self.backend = "json"
        self.last_error = ""

        ensure_json_file(self.storage_path, [])

        if mongodb_uri:
            self._connect_mongo()

    def _connect_mongo(self) -> None:
        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=3000)
            self.client.admin.command("ping")
            database = self.client["bugfix_prompt_arena"]
            self.collection = database["submissions"]
            self.backend = "mongodb"
            self.last_error = ""
        except PyMongoError as exc:
            self.client = None
            self.collection = None
            self.backend = "json"
            self.last_error = str(exc)

    def save_submission(self, submission: dict[str, Any]) -> str:
        if self.collection is not None:
            try:
                self.collection.insert_one(submission)
                self.backend = "mongodb"
                return self.backend
            except PyMongoError as exc:
                self.last_error = str(exc)
                self.backend = "json"

        records = self.fetch_submissions()
        records.append(submission)
        with self.storage_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, ensure_ascii=False)
        return self.backend

    def fetch_submissions(self) -> list[dict]:
        if self.collection is not None:
            try:
                items = list(self.collection.find({}, {"_id": 0}).sort("created_at", -1))
                self.backend = "mongodb"
                return items
            except PyMongoError as exc:
                self.last_error = str(exc)
                self.backend = "json"

        ensure_json_file(self.storage_path, [])
        try:
            with self.storage_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def status_message(self) -> str:
        if self.backend == "mongodb":
            return "MongoDB connected"
        if self.mongodb_uri:
            return f"Using local JSON fallback ({self.last_error or 'MongoDB unavailable'})"
        return "Using local JSON fallback (MONGODB_URI not configured)"
