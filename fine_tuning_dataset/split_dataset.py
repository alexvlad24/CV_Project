import os
import json
import sys
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from huggingface_hub import login
from resumeitems import ResumeItem

sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv(override=True)
hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    raise ValueError("❌ 'HF_TOKEN' not found in .env file!")

login(token=hf_token)

HF_DATASET_NAME = "alecs-vlad24/cv-resume-structuring-v2pro"


# Writes a list of dictionary records into a JSONL file format.
def save_jsonl(filename: str, dataset: list) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# Performs stratified dataset splitting into train, validation, and test sets, and uploads them to Hugging Face Hub.
def split_and_upload_dataset(
    input_jsonl_path: str = "jsonl/resumes_for_split.jsonl",
) -> None:
    data = []
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    labels = [
        item.get("category") or item.get("Category") or "Unknown" for item in data
    ]

    print(f"Total loaded examples: {len(data)}")
    print(f"Primary category distribution: {Counter(labels).most_common(5)}")

    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, test_size=0.20, stratify=labels, random_state=42
    )

    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )

    print("\n✅ Dataset split completed:")
    print(
        f" - Train set:       {len(train_data)} examples ({len(train_data) / len(data):.1%})"
    )
    print(
        f" - Validation set:  {len(val_data)} examples ({len(val_data) / len(data):.1%})"
    )
    print(
        f" - Test set:        {len(test_data)} examples ({len(test_data) / len(data):.1%})"
    )

    save_jsonl("fine_tuning_dataset/train.jsonl", train_data)
    save_jsonl("fine_tuning_dataset/val.jsonl", val_data)
    save_jsonl("fine_tuning_dataset/test.jsonl", test_data)
    print(
        "💾 Local files 'train.jsonl', 'val.jsonl', and 'test.jsonl' saved successfully!\n"
    )

    print("🔄 Converting dictionaries to ResumeItem instances...")
    train_items = [ResumeItem.model_validate(item) for item in train_data]
    val_items = [ResumeItem.model_validate(item) for item in val_data]
    test_items = [ResumeItem.model_validate(item) for item in test_data]

    print(f"🚀 Uploading dataset splits to Hugging Face Hub: {HF_DATASET_NAME}...")
    ResumeItem.push_to_hub(HF_DATASET_NAME, train_items, val_items, test_items)
    print("🎉 Dataset successfully published on Hugging Face Hub.")


if __name__ == "__main__":
    split_and_upload_dataset()
