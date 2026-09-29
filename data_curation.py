import json
import re


# Sanitizes input text by removing line breaks, escape sequences, non-ASCII characters, and repetitive separator lines.
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


# Filters, sanitizes, truncates raw resume entries based on character thresholds, and exports cleaned data to a JSONL file.
def clean_to_essentials(
    input_path: str, output_path: str, max_chars: int = 5000, min_chars: int = 300
):
    print(f"🧹 Sanitizing text and truncating content to MAX {max_chars} characters...")

    cleaned_records = []
    total_processed = 0
    skipped_too_short = 0
    truncated_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                total_processed += 1
                data = json.loads(line)

                raw_text = data.get("Text", "")
                category = data.get("Category", "IT Professional")

                cleaned_text = clean_text_advanced(raw_text)

                if len(cleaned_text) < min_chars:
                    print(line)
                    skipped_too_short += 1
                    continue

                if len(cleaned_text) > max_chars:
                    cleaned_text = cleaned_text[:max_chars].strip()
                    truncated_count += 1

                clean_item = {"Category": category, "Text": cleaned_text}

                cleaned_records.append(clean_item)

            except json.JSONDecodeError:
                pass

    print("\n" + "=" * 50)
    print("📊 DATASET SANITIZATION & OPTIMIZATION REPORT")
    print("=" * 50)
    print(f"🔹 Total records processed: {total_processed:,}")
    print(f"✅ Valid records preserved: {len(cleaned_records):,}")
    print(f"✂️ Records truncated to {max_chars} chars: {truncated_count:,}")
    print(f"🗑️ Skipped short records (< {min_chars} chars): {skipped_too_short:,}")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in cleaned_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n📁 Cleaned file saved to: {output_path}\n")

    print("=" * 60)
    print("📝 SAMPLE RECORD PREVIEW:")
    print("=" * 60)
    if cleaned_records:
        sample = cleaned_records[0]
        print(f"Category: {sample['Category']}")
        print(f"Text Length: {len(sample['Text'])} characters")
        print(f"Text Preview:\n{sample['Text'][:400]}...")


if __name__ == "__main__":
    clean_to_essentials(
        input_path="resumes_dataset.jsonl",
        output_path="resumes_step1_cleaned2.jsonl",
        max_chars=5000,
        min_chars=300,
    )
