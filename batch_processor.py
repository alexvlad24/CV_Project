import json
import os
import pickle
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from resumeitems import ResumeItem, SYSTEM_PROMPT

load_dotenv(override=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

MODEL = "gpt-4o-mini"
BATCHES_FOLDER = "batches/new"
OUTPUT_FOLDER = "output/new"
STATE_FILE = Path("resume_batches_v2.pkl")


class ResumeBatchProcessor:
    BATCH_SIZE = 100
    batches: List["ResumeBatchProcessor"] = []

    # Initializes a batch chunk instance for a slice of resume items.
    def __init__(
        self, items: List[ResumeItem], start: int, end: int, lite: bool = True
    ):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"resumes_{start}_{end}.jsonl"
        self.file_id: Optional[str] = None
        self.batch_id: Optional[str] = None
        self.output_file_id: Optional[str] = None
        self.done = False

        folder = Path("lite") if lite else Path("full")
        self.batches_dir = folder / BATCHES_FOLDER
        self.output_dir = folder / OUTPUT_FOLDER

        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Formats a single resume item into an OpenAI Batch API compatible JSONL line.
    def _make_jsonl_line(self, item: ResumeItem, index_id: int) -> str:
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{item.text}"},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        line = {
            "custom_id": str(index_id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(line)

    # Writes all assigned resume items to a local JSONL batch file.
    def make_file(self):
        batch_file_path = self.batches_dir / self.filename
        with batch_file_path.open("w", encoding="utf-8") as f:
            for idx in range(self.start, self.end):
                f.write(self._make_jsonl_line(self.items[idx], idx))
                f.write("\n")

    # Uploads the batch JSONL file to OpenAI Files storage.
    def send_file(self):
        if not client:
            raise ValueError("OPENAI_API_KEY is not set in the .env file!")
        batch_file_path = self.batches_dir / self.filename
        with batch_file_path.open("rb") as f:
            response = client.files.create(file=f, purpose="batch")
        self.file_id = response.id

    # Submits the uploaded batch file to the OpenAI Batch endpoint.
    def submit_batch(self):
        if not client:
            raise ValueError("OPENAI_API_KEY is not set in the .env file!")
        response = client.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )
        self.batch_id = response.id

    # Checks whether the batch job processing has completed on OpenAI.
    def is_ready(self) -> bool:
        if not client:
            raise ValueError("OPENAI_API_KEY is not set in the .env file!")
        response = client.batches.retrieve(self.batch_id)
        status = response.status
        if status == "completed":
            self.output_file_id = response.output_file_id
            return True
        return False

    # Downloads the processed batch output content to the local output folder.
    def fetch_output(self):
        if not client:
            raise ValueError("OPENAI_API_KEY is not set in the .env file!")
        output_file_path = str(self.output_dir / self.filename)
        response = client.files.content(self.output_file_id)
        response.write_to_file(output_file_path)

    # Parses output JSONL results and assigns extracted fields to corresponding ResumeItems.
    def apply_output(self):
        output_file_path = self.output_dir / self.filename
        with output_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                json_line = json.loads(line)
                item_id = int(json_line["custom_id"])

                raw_response = json_line["response"]["body"]["choices"][0]["message"][
                    "content"
                ]

                try:
                    data = json.loads(raw_response)
                    target_item = self.items[item_id]

                    target_item.candidate_role = data.get("Candidate_Role") or data.get(
                        "candidate_role"
                    )
                    target_item.experience_level = data.get(
                        "Experience_Level"
                    ) or data.get("experience_level")
                    target_item.years_of_experience = data.get(
                        "Years_Of_Experience"
                    ) or data.get("years_of_experience")
                    target_item.primary_skills = data.get("Primary_Skills") or data.get(
                        "primary_skills", []
                    )
                    target_item.clean_summary = data.get("Clean_Summary") or data.get(
                        "clean_summary"
                    )

                    target_item.build_sft_messages()

                except json.JSONDecodeError:
                    print(f"⚠️ Warning: Invalid JSON response for item ID {item_id}")

        self.done = True

    # Splits the list of ResumeItems into batch processing chunks.
    @classmethod
    def create(cls, items: List[ResumeItem], lite: bool = True):
        cls.batches = []
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = ResumeBatchProcessor(items, start, end, lite)
            cls.batches.append(batch)
        print(f"📦 Created {len(cls.batches)} batch(es).")

    # Creates files, uploads, and submits all batch chunks to OpenAI.
    @classmethod
    def run(cls):
        for batch in tqdm(cls.batches, desc="🚀 Submitting batches to OpenAI"):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"✅ Submitted {len(cls.batches)} batch(es) to OpenAI.")

    # Downloads output for completed batches and applies results to in-memory items.
    @classmethod
    def fetch(cls):
        for batch in tqdm(cls.batches, desc="📥 Fetching results"):
            if not batch.done:
                if batch.is_ready():
                    batch.fetch_output()
                    batch.apply_output()

        finished = [b for b in cls.batches if b.done]
        print(f"🏁 Finished: {len(finished)} of {len(cls.batches)} batch(es).")

    # Serializes the batch processor state to disk.
    @classmethod
    def save_state(cls):
        items_backup = cls.batches[0].items if cls.batches else None
        for batch in cls.batches:
            batch.items = None

        with STATE_FILE.open("wb") as f:
            pickle.dump(cls.batches, f)

        for batch in cls.batches:
            batch.items = items_backup
        print("💾 Batch processor state saved to disk.")

    # Deserializes the batch processor state from disk.
    @classmethod
    def load_state(cls, items: List[ResumeItem]):
        with STATE_FILE.open("rb") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
        print(f"📂 Loaded {len(cls.batches)} previously saved batch(es).")
