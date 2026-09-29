import json
import re
import os
from datetime import datetime
from typing import List, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from resumeitems import ResumeItem

CHUNK_SIZE = 500
WORKERS = max((os.cpu_count() or 2) - 1, 1)


class ResumeLoader:
    # Initializes the loader with input path and string length thresholds.
    def __init__(self, input_path: str, max_chars: int = 5000, min_chars: int = 300):
        self.input_path = input_path
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.raw_data: List[dict] = []

    # Sanitizes raw text by removing non-standard characters, excessive whitespaces, and repetitive punctuation artifacts.
    @staticmethod
    def clean_text_advanced(text: str) -> str:
        text = (
            text.replace("\\n", " ")
            .replace("\n", " ")
            .replace("\t", " ")
            .replace("\\t", " ")
        )
        text = re.sub(r"[^\w\s.,;:/\-+#@()&]", "", text)
        text = re.sub(r"={2,}", "", text)
        text = re.sub(r"-{2,}", "", text)
        text = re.sub(r"_{2,}", "", text)
        return re.sub(r"\s+", " ", text).strip()

    # Validates, cleans, and converts a raw JSON dictionary entry into a structured ResumeItem instance.
    def parse_datapoint(self, datapoint: dict) -> Optional[ResumeItem]:
        raw_text = datapoint.get("Text", "")
        category = datapoint.get("Category", "IT Professional")

        cleaned = self.clean_text_advanced(raw_text)

        if len(cleaned) < self.min_chars:
            return None

        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars].strip()

        return ResumeItem(category=category, text=cleaned)

    # Processes a single batch chunk of raw resume records and filters out invalid entries.
    def parse_chunk(self, chunk: List[dict]) -> List[ResumeItem]:
        results = [self.parse_datapoint(dp) for dp in chunk]
        return [item for item in results if item is not None]

    # Yields consecutive chunks of raw resume data for multiprocessing pools.
    def chunk_generator(self):
        for i in range(0, len(self.raw_data), CHUNK_SIZE):
            yield self.raw_data[i : i + CHUNK_SIZE]

    # Loads raw JSONL lines and parses them into ResumeItem instances using multi-core parallel processing.
    def load(self, workers: int = WORKERS) -> List[ResumeItem]:
        start = datetime.now()
        print(f"📖 Reading file: {self.input_path}...")

        with open(self.input_path, "r", encoding="utf-8") as f:
            self.raw_data = [json.loads(line) for line in f if line.strip()]

        print(
            f"⚡ Processing {len(self.raw_data):,} resumes using {workers} CPU workers..."
        )

        results: List[ResumeItem] = []
        chunk_count = (len(self.raw_data) // CHUNK_SIZE) + 1

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(
                pool.map(self.parse_chunk, self.chunk_generator()), total=chunk_count
            ):
                results.extend(batch)

        finish = datetime.now()
        duration = (finish - start).total_seconds()
        print(
            f"✅ Successfully extracted {len(results):,} valid ResumeItem objects in {duration:.2f} seconds!\n"
        )
        return results
