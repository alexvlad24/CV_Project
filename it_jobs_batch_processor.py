import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv(override=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

MODEL = "gpt-4o-mini"
BATCHES_FOLDER = "job_batches"
OUTPUT_FOLDER = "job_output"
STATE_FILE = Path("job_batches_state.pkl")

SYSTEM_PROMPT = """You are an expert IT Talent Acquisition Specialist and Senior Systems Architect.
Your task is to generate realistic, detailed, and highly technical job descriptions for a given IT role across 3 distinct experience levels: Junior, Mid-Level, and Senior/Lead.

For the requested IT role, generate a JSON object containing a "jobs" array with EXACTLY 3 objects (one for each seniority level).

Each job object in the array MUST strictly follow this JSON structure:
{
  "title": "<Exact Job Title>",
  "experience_level": "Junior" | "Mid" | "Senior",
  "years_of_experience": "<e.g. 0-2 years | 2-5 years | 5+ years>",
  "primary_tech_stack": ["<Tech1>", "<Tech2>", "<Tech3>", ...],
  "required_skills": ["<Skill1>", "<Skill2>", "<Methodology/Tool>", ...],
  "responsibilities": ["<Responsibility 1>", "<Responsibility 2>", "<Responsibility 3>"],
  "domain_taxonomy": "<Domain Category, e.g., Cloud & DevOps | Software Engineering | Data & AI | Cybersecurity | Frontend & Mobile>"
}

CRITICAL RULES:
1. Ensure the primary_tech_stack and required_skills reflect actual modern IT market demands.
2. Differentiate clearly between levels:
   - Junior: Focus on foundational concepts, core languages, basic tools, bug fixing, and learning.
   - Mid: Focus on production-level frameworks, cloud services, automated testing, and CI/CD pipelines.
   - Senior: Focus on system architecture, scalable design, security, mentoring, and multi-cloud / advanced infrastructure.
3. Do NOT include generic non-technical HR filler text (like "great company culture" or "free coffee"). Keep it 100% focused on technical requirements and core duties.
4. Avoid generic terms like 'scripting skills' or 'cloud concepts'; always specify concrete tools or framework names.
5. Always use modern technology stacks for every job.
6. Maintain a logical tech stack progression across levels (e.g., Junior builds components, Mid handles state/CI-CD, Senior architectures distributed systems). Do not randomly switch language ecosystems between levels.
7. Match technologies strictly to the specific role.
8. MANDATORY ROLE ANCHORS: Include the primary defining technology/framework of the requested role across ALL 3 seniority levels (e.g., 'dbt' for Analytics Engineer, 'Terraform' for IaC roles, 'Kubernetes' for Cloud/DevOps). Do not delay the core stack to Senior level.
9. Output MUST be valid JSON only.
"""


class ITJobRoleItem:
    # Initializes a single IT job role data container.
    def __init__(self, role_id: int, role_name: str):
        self.role_id = role_id
        self.role_name = role_name


class ITJobBatchProcessor:
    BATCH_SIZE = 12
    batches: List["ITJobBatchProcessor"] = []

    # Initializes a batch chunk instance with start and end index bounds.
    def __init__(self, items: List[ITJobRoleItem], start: int, end: int):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"jobs_batch_{start}_{end}.jsonl"
        self.file_id: Optional[str] = None
        self.batch_id: Optional[str] = None
        self.output_file_id: Optional[str] = None
        self.done = False

        self.batches_dir = Path(BATCHES_FOLDER)
        self.output_dir = Path(OUTPUT_FOLDER)

        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Formats a single IT job role item into an OpenAI Batch API compatible JSONL line.
    def _make_jsonl_line(self, item: ITJobRoleItem) -> str:
        user_prompt = f"Generate 3 job description variants (Junior, Mid, Senior) for the IT role: '{item.role_name}'."

        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        line = {
            "custom_id": f"role_{item.role_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(line)

    # Writes all assigned job role items into a local JSONL batch file.
    def make_file(self):
        batch_file_path = self.batches_dir / self.filename
        with batch_file_path.open("w", encoding="utf-8") as f:
            for idx in range(self.start, self.end):
                f.write(self._make_jsonl_line(self.items[idx]))
                f.write("\n")

    # Uploads the generated batch JSONL file to OpenAI Files storage.
    def send_file(self):
        if not client:
            raise ValueError("OPENAI_API_KEY is not set in the .env file!")
        batch_file_path = self.batches_dir / self.filename
        with batch_file_path.open("rb") as f:
            response = client.files.create(file=f, purpose="batch")
        self.file_id = response.id

    # Submits the uploaded batch file to the OpenAI Batch processing endpoint.
    def submit_batch(self):
        if not client:
            raise ValueError("OPENAI_API_KEY is not set in the .env file!")
        response = client.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )
        self.batch_id = response.id

    # Checks whether the batch job processing on OpenAI has completed.
    def is_ready(self) -> bool:
        if not client:
            raise ValueError("OPENAI_API_KEY is not set in the .env file!")
        response = client.batches.retrieve(self.batch_id)
        status = response.status
        if status == "completed":
            self.output_file_id = response.output_file_id
            return True
        elif status in ["failed", "canceled", "expired"]:
            print(f"❌ Batch {self.batch_id} failed with status: {status}")
        return False

    # Downloads the processed batch output file content to the local output folder.
    def fetch_output(self):
        if not client:
            raise ValueError("OPENAI_API_KEY is not set in the .env file!")
        output_file_path = self.output_dir / self.filename
        response = client.files.content(self.output_file_id)

        with output_file_path.open("wb") as f:
            f.write(response.content)

    # Parses the downloaded JSONL result lines into a list of structured job dictionaries.
    def parse_results(self) -> List[Dict[str, Any]]:
        extracted_jobs = []
        output_file_path = self.output_dir / self.filename

        if not output_file_path.exists():
            return []

        with output_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                json_line = json.loads(line)
                custom_id = json_line["custom_id"]
                raw_response = json_line["response"]["body"]["choices"][0]["message"][
                    "content"
                ]

                try:
                    data = json.loads(raw_response)
                    jobs_list = (
                        data.get("jobs", data) if isinstance(data, dict) else data
                    )

                    if isinstance(jobs_list, list):
                        for job in jobs_list:
                            extracted_jobs.append(job)
                    else:
                        print(f"⚠️ Warning: Unexpected JSON structure for {custom_id}")

                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON response for {custom_id}")

        self.done = True
        return extracted_jobs

    # Partitions the full list of role items into multiple batch processor chunks.
    @classmethod
    def create(cls, items: List[ITJobRoleItem]):
        cls.batches = []
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = ITJobBatchProcessor(items, start, end)
            cls.batches.append(batch)
        print(f"📦 Created {len(cls.batches)} batch file(s) for OpenAI.")

    # Generates, uploads, and submits all batch chunks to the OpenAI API.
    @classmethod
    def run(cls):
        for batch in tqdm(
            cls.batches, desc="🚀 Submitting batches to OpenAI Batch API"
        ):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"✅ Successfully submitted {len(cls.batches)} batch(es) to OpenAI.")

    # Downloads completed batch outputs, parses job descriptions, and saves them to a CSV file.
    @classmethod
    def fetch_and_save_csv(cls, output_csv_filename: str = "it_jobs_synthetic_279.csv"):
        all_extracted_jobs = []
        for batch in tqdm(cls.batches, desc="📥 Fetching and processing results"):
            if not batch.done:
                if batch.is_ready():
                    batch.fetch_output()
                    jobs = batch.parse_results()
                    all_extracted_jobs.extend(jobs)
                else:
                    print(f"⌛ Batch {batch.batch_id} is still processing on OpenAI...")
            else:
                jobs = batch.parse_results()
                all_extracted_jobs.extend(jobs)

        if all_extracted_jobs:
            df = pd.DataFrame(all_extracted_jobs)
            df.to_csv(output_csv_filename, index=False)
            print(
                f"🎉 SUCCESS! Saved {len(df)} job descriptions into '{output_csv_filename}'"
            )
        else:
            print("⚠️ No completed jobs fetched yet. Run fetch again later.")

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
    def load_state(cls, items: List[ITJobRoleItem]):
        if STATE_FILE.exists():
            with STATE_FILE.open("rb") as f:
                cls.batches = pickle.load(f)
            for batch in cls.batches:
                batch.items = items
            print(f"📂 Loaded {len(cls.batches)} previously saved batch(es).")
        else:
            print("⚠️ No state file found. Create new batches first.")
