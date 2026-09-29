import json
import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
)
from fastembed import SparseTextEmbedding

load_dotenv(override=True)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

print("⏳ Loading local BM25 model (fastembed)...")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

qdrant_client = QdrantClient(path="qdrant_db")

COLLECTION_NAME = "it_jobs"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536

JSONL_FILE_PATH = "job_output/combined_responses.jsonl"


# Reads a JSONL file and extracts all individual job description objects.
def parse_jobs_from_jsonl(file_path: str):
    all_jobs = []
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"❌ JSONL file not found at path: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)

                if "response" in data and "body" in data["response"]:
                    raw_content = data["response"]["body"]["choices"][0]["message"][
                        "content"
                    ]
                    parsed_content = json.loads(raw_content)
                    jobs = parsed_content.get("jobs", parsed_content)
                elif "jobs" in data:
                    jobs = data["jobs"]
                elif isinstance(data, dict) and "title" in data:
                    jobs = [data]
                else:
                    jobs = data

                if isinstance(jobs, list):
                    all_jobs.extend(jobs)
                else:
                    print(f"⚠️ Unexpected structure at line {line_idx + 1}")

            except Exception as e:
                print(f"⚠️ Error processing line {line_idx + 1}: {e}")

    return all_jobs


# Concatenates job fields into a structured text format suitable for dense and sparse embeddings.
def build_dense_payload(job: dict) -> str:
    tech_stack = ", ".join(job.get("primary_tech_stack", []))
    skills = ", ".join(job.get("required_skills", []))
    responsibilities = "; ".join(job.get("responsibilities", []))

    payload_text = (
        f"Job Title: {job.get('title', 'N/A')} | "
        f"Seniority Level: {job.get('experience_level', 'N/A')} ({job.get('years_of_experience', 'N/A')}) | "
        f"Domain: {job.get('domain_taxonomy', 'N/A')} | "
        f"Primary Tech Stack: {tech_stack} | "
        f"Required Skills: {skills} | "
        f"Responsibilities: {responsibilities}"
    )
    return payload_text


# Indexes the job descriptions into Qdrant using dense and BM25 sparse vectors.
def main():
    print(f"📂 Reading and parsing data from: '{JSONL_FILE_PATH}'...")
    jobs = parse_jobs_from_jsonl(JSONL_FILE_PATH)
    print(f"✅ Extracted {len(jobs)} job descriptions from JSONL file.")

    collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"🗑️ Deleting existing collection '{COLLECTION_NAME}'...")
        qdrant_client.delete_collection(COLLECTION_NAME)

    print(f"🏗️ Creating hybrid collection '{COLLECTION_NAME}' (Dense + BM25 Sparse)...")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        },
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )

    points = []
    print("🧠 Generating Dense (OpenAI) and Sparse BM25 (FastEmbed) vectors...")

    for idx, job in tqdm(
        enumerate(jobs), total=len(jobs), desc="Processing Hybrid Jobs"
    ):
        dense_text = build_dense_payload(job)

        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL, input=dense_text
        )
        dense_vector = response.data[0].embedding

        sparse_embeddings = list(sparse_model.embed([dense_text]))[0]
        sparse_vector = SparseVector(
            indices=sparse_embeddings.indices.tolist(),
            values=sparse_embeddings.values.tolist(),
        )

        payload = {
            "title": job.get("title", ""),
            "experience_level": job.get("experience_level", ""),
            "years_of_experience": job.get("years_of_experience", ""),
            "domain_taxonomy": job.get("domain_taxonomy", ""),
            "primary_tech_stack": job.get("primary_tech_stack", []),
            "required_skills": job.get("required_skills", []),
            "responsibilities": job.get("responsibilities", []),
            "dense_payload": dense_text,
        }

        point = PointStruct(
            id=idx,
            vector={"dense": dense_vector, "sparse": sparse_vector},
            payload=payload,
        )
        points.append(point)

    print(f"🚀 Uploading {len(points)} hybrid points to Qdrant...")
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

    print("\n" + "=" * 70)
    print(
        "🎉 SUCCESS! Qdrant database populated with hybrid indexing (Dense + BM25 Sparse)!"
    )
    print("📁 Saved locally at: 'qdrant_db/'")
    print("=" * 70)


if __name__ == "__main__":
    main()
