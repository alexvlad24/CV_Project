import json
import re
from tqdm import tqdm


# Calculates the Jaccard similarity index between two skill lists.
def jaccard_similarity(list1, list2):
    if not list1 or not list2:
        return 0.0
    s1 = set([str(x).lower().strip() for x in list1])
    s2 = set([str(x).lower().strip() for x in list2])
    if not s1 and not s2:
        return 1.0
    return len(s1.intersection(s2)) / len(s1.union(s2))


# Safely parses JSON strings, extracting raw blocks or markdown-wrapped JSON objects.
def parse_json_safely(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass
    return {}


# Evaluates model predictions against ground-truth CV data measuring JSON validity, role match, and skill overlap.
def evaluate(predict_fn, test_data, max_examples=None):
    data_to_eval = (
        test_data if max_examples is None else test_data.select(range(max_examples))
    )
    size = len(data_to_eval)

    json_valid_count = 0
    skill_scores = []
    role_matches = 0

    print(f"\n🚀 Starting evaluation on {size} resume examples...")

    for i in tqdm(range(size), desc="Running Inference"):
        item = data_to_eval[i]

        ground_truth_text = [m for m in item["messages"] if m["role"] == "assistant"][
            0
        ]["content"]
        truth_json = parse_json_safely(ground_truth_text)

        truth_role = (
            str(
                truth_json.get("Candidate_Role")
                or truth_json.get("candidate_role")
                or ""
            )
            .lower()
            .strip()
        )

        truth_skills = (
            truth_json.get("Primary_Skills") or truth_json.get("primary_skills") or []
        )

        response_text = predict_fn(item)
        predicted_json = parse_json_safely(response_text)

        if bool(predicted_json):
            json_valid_count += 1

        pred_role = (
            str(
                predicted_json.get("Candidate_Role")
                or predicted_json.get("candidate_role")
                or ""
            )
            .lower()
            .strip()
        )

        pred_skills = (
            predicted_json.get("Primary_Skills")
            or predicted_json.get("primary_skills")
            or []
        )

        if (
            truth_role
            and pred_role
            and ((truth_role in pred_role) or (pred_role in truth_role))
        ):
            role_matches += 1

        skill_scores.append(jaccard_similarity(truth_skills, pred_skills))

    avg_skill = (sum(skill_scores) / size) * 100 if size > 0 else 0
    valid_rate = (json_valid_count / size) * 100
    role_acc = (role_matches / size) * 100

    print("\n" + "=" * 55)
    print("📊 EVALUATION RESULTS: FINE-TUNED MODEL")
    print("=" * 55)
    print(f"✅ Valid JSON Format:        {valid_rate:.1f}%")
    print(f"🎯 Role Match Accuracy:      {role_acc:.1f}%")
    print(f"💡 Skill Overlap (Jaccard):  {avg_skill:.1f}%")
    print("=" * 55)
