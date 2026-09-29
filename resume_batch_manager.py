import json
import time
from pathlib import Path
from loaders import ResumeLoader
from resumeitems import ResumeItem
from batch_processor import ResumeBatchProcessor, STATE_FILE
from openai import OpenAI

LITE_MODE = True
MAX_CONCURRENT_BATCHES = 3


# Serializes a list of ResumeItem objects into a UTF-8 encoded JSONL file.
def save_items_to_jsonl(items: list[ResumeItem], file_path: str | Path) -> None:
    path = Path(file_path)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(), ensure_ascii=False))
            f.write("\n")
    print(f"💾 Successfully saved to {path.name}")


# Orchestrates rate-limited batch submission, remote status tracking, and final dataset compilation.
def main():
    raw_jsonl_file = "jsonl/resumes_step1_cleaned2.jsonl"
    curated_output_file = "jsonl/resumes_curated_completed2.jsonl"

    print("--- STEP 1: Loading Raw Data ---")
    loader = ResumeLoader(input_path=raw_jsonl_file)
    items = loader.load()

    print("\n--- STEP 2: Preparing / Loading Batches ---")
    ResumeBatchProcessor.BATCH_SIZE = 100

    if STATE_FILE.exists():
        print("📂 State file found. Loading existing batches...")
        ResumeBatchProcessor.load_state(items)
    else:
        print("🆕 No state file found. Creating batch infrastructure...")
        ResumeBatchProcessor.create(items, lite=LITE_MODE)

    client = OpenAI()

    print("\n🔍 Synchronizing remote batch statuses with OpenAI...")
    for b in ResumeBatchProcessor.batches:
        if getattr(b, "batch_id", None) and not b.done:
            try:
                remote_batch = client.batches.retrieve(b.batch_id)
                status = remote_batch.status

                if status == "completed":
                    print(f"✅ Batch {b.filename} is completed. Downloading results...")
                    b.output_file_id = remote_batch.output_file_id
                    b.fetch_output()
                    b.apply_output()
                elif status in ["failed", "cancelled", "expired"]:
                    print(
                        f"⚠️ Batch {b.filename} (ID: {b.batch_id}) ended with '{status}'. Resetting for re-submission..."
                    )
                    b.batch_id = None
                    b.file_id = None
            except Exception as e:
                print(f"⚠️ Error verifying batch {b.filename}: {e}")

    ResumeBatchProcessor.save_state()

    active_batches = [
        b
        for b in ResumeBatchProcessor.batches
        if getattr(b, "batch_id", None) is not None and not b.done
    ]
    unsubmitted = [
        b
        for b in ResumeBatchProcessor.batches
        if getattr(b, "batch_id", None) is None and not b.done
    ]
    completed_batches = [b for b in ResumeBatchProcessor.batches if b.done]

    print(
        f"\n📊 PROCESSING STATUS: Total: {len(ResumeBatchProcessor.batches)} | Completed: {len(completed_batches)} | Active: {len(active_batches)} | Pending: {len(unsubmitted)}"
    )

    slots_available = MAX_CONCURRENT_BATCHES - len(active_batches)

    if slots_available > 0 and unsubmitted:
        to_submit = unsubmitted[:slots_available]
        print(
            f"\n🚀 Submitting {len(to_submit)} new batches (available slots: {slots_available})..."
        )

        for b in to_submit:
            try:
                print(f"   -> Submitting {b.filename}...")
                b.make_file()
                b.send_file()
                b.submit_batch()
                print(f"      ✅ Submitted! Batch ID: {b.batch_id}")
            except Exception as e:
                print(f"   ❌ Error submitting batch {b.filename}: {e}")

        ResumeBatchProcessor.save_state()
        print("\n💾 State saved. New batches are processing.")

    still_pending = [b for b in ResumeBatchProcessor.batches if not b.done]

    if still_pending:
        print(f"\n⏳ In Progress: {len(still_pending)} batches remaining.")
        print(
            "💡 Re-run this script later to fetch completed results and submit next batch wave."
        )
        return

    print("\n--- STEP 3: Saving Final Curated Dataset ---")
    save_items_to_jsonl(items, curated_output_file)
    print(f"\n🏆 PROCESS COMPLETED SUCCESSFULLY! All {len(items)} resumes processed.")


if __name__ == "__main__":
    main()
