import json
from pathlib import Path
from resumeitems import ResumeItem

RAW_STEP1_FILE = Path("jsonl/resumes_step1_cleaned2.jsonl")
OUTPUT_DIR = Path("jsonl/resumes_output.jsonl")
FINAL_OUTPUT = Path("jsonl/resumes_for_split.jsonl")


# Merges raw resume inputs with asynchronous LLM batch responses and exports the final SFT dataset to JSONL.
def reconstruct_dataset() -> None:
    raw_requests = {}
    print(f"📖 Reading raw inputs from {RAW_STEP1_FILE}...")

    with RAW_STEP1_FILE.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = str(idx)

            raw_requests[custom_id] = {
                "text": (item.get("text") or item.get("Text") or "").strip(),
                "category": item.get("category") or item.get("Category") or "Unknown",
            }

    print(f"✅ Loaded {len(raw_requests)} raw records with original categories.")

    response_files = (
        sorted(OUTPUT_DIR.glob("*.jsonl")) if OUTPUT_DIR.is_dir() else [OUTPUT_DIR]
    )
    print(f"📥 Reading LLM responses from {OUTPUT_DIR}...")

    completed_items = []

    for r_file in response_files:
        with r_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                resp_obj = json.loads(line)
                custom_id = str(resp_obj.get("custom_id"))

                if custom_id in raw_requests:
                    try:
                        content_str = resp_obj["response"]["body"]["choices"][0][
                            "message"
                        ]["content"]
                        extracted_data = json.loads(content_str)
                    except (KeyError, json.JSONDecodeError):
                        extracted_data = resp_obj

                    raw_info = raw_requests[custom_id]

                    item = ResumeItem(
                        category=raw_info["category"],
                        text=raw_info["text"],
                        candidate_role=extracted_data.get("Candidate_Role")
                        or extracted_data.get("candidate_role"),
                        experience_level=extracted_data.get("Experience_Level")
                        or extracted_data.get("experience_level"),
                        years_of_experience=(
                            extracted_data.get("Years_Of_Experience")
                            or extracted_data.get("years_of_experience")
                            or extracted_data.get("Years_of_Experience")
                        ),
                        primary_skills=extracted_data.get("Primary_Skills")
                        or extracted_data.get("primary_skills", []),
                        clean_summary=extracted_data.get("Clean_Summary")
                        or extracted_data.get("clean_summary"),
                    )

                    item.build_sft_messages()
                    completed_items.append(item.model_dump())

    print(
        f"✨ Successfully matched and reconstructed {len(completed_items)} complete training samples."
    )

    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with FINAL_OUTPUT.open("w", encoding="utf-8") as out_f:
        for item_dict in completed_items:
            out_f.write(json.dumps(item_dict, ensure_ascii=False) + "\n")

    print(f"🏆 Final dataset successfully saved to '{FINAL_OUTPUT}'!")


if __name__ == "__main__":
    reconstruct_dataset()
