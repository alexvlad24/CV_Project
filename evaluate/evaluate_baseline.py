import json
from litellm import completion
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from resumeitems import ResumeItem, SYSTEM_PROMPT


# Calculates the Jaccard similarity index between two skill lists.
def jaccard_similarity(list1, list2):
    if not list1 or not list2:
        return 0.0
    s1, s2 = set([str(x).lower() for x in list1]), set([str(x).lower() for x in list2])
    if not s1 and not s2:
        return 1.0
    return len(s1.intersection(s2)) / len(s1.union(s2))


# Sends resume text to a local Ollama Llama 3.2 instance and returns the parsed JSON response.
def ollama_baseline_predictor(item: ResumeItem) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Extract structured information from this resume:\n\n{item.text}",
        },
    ]
    try:
        response = completion(
            model="ollama/llama3.2",
            messages=messages,
            api_base="http://localhost:11434",
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception:
        return {}


class BaselineEvaluator:
    # Initializes the evaluation harness with predictor function, test dataset, sample size, and thread count.
    def __init__(self, predictor, test_data, size=50, workers=2):
        self.predictor = predictor
        self.test_data = test_data[:size]
        self.size = len(self.test_data)
        self.workers = workers

        self.json_valid_count = 0
        self.skill_scores = []
        self.role_matches = 0

    # Evaluates a single test resume against the ground-truth annotations.
    def evaluate_single(self, i):
        item = self.test_data[i]
        predicted_json = self.predictor(item)

        is_valid = bool(predicted_json)
        if not is_valid:
            return False, 0.0, False

        truth_role = str(item.candidate_role or "").lower()
        truth_skills = item.primary_skills or []

        pred_role = str(
            predicted_json.get("Candidate_Role")
            or predicted_json.get("candidate_role")
            or ""
        ).lower()
        pred_skills = (
            predicted_json.get("Primary_Skills")
            or predicted_json.get("primary_skills")
            or []
        )

        role_match = (
            (truth_role in pred_role) or (pred_role in truth_role)
            if truth_role and pred_role
            else False
        )
        skill_score = jaccard_similarity(truth_skills, pred_skills)

        return True, skill_score, role_match

    # Runs concurrent baseline evaluation across the specified test split.
    def run(self):
        print(
            f"\n🚀 Evaluating OLLAMA (Untrained Llama 3.2 Baseline) on {self.size} test examples..."
        )

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            results = list(
                tqdm(ex.map(self.evaluate_single, range(self.size)), total=self.size)
            )

        for valid, skill_score, role_match in results:
            if valid:
                self.json_valid_count += 1
            self.skill_scores.append(skill_score)
            if role_match:
                self.role_matches += 1

        self.report()

    # Computes and prints aggregate evaluation metrics to the console.
    def report(self):
        avg_skill_score = (
            (sum(self.skill_scores) / self.size) * 100 if self.size > 0 else 0
        )
        valid_rate = (self.json_valid_count / self.size) * 100
        role_acc = (self.role_matches / self.size) * 100

        print("\n" + "=" * 55)
        print("📊 BASELINE EVALUATION RESULTS (UNTRAINED LLAMA 3.2 - OLLAMA)")
        print("=" * 55)
        print(f"✅ Valid JSON Format:        {valid_rate:.1f}%")
        print(f"🎯 Role Match Accuracy:      {role_acc:.1f}%")
        print(f"💡 Skill Overlap (Jaccard):  {avg_skill_score:.1f}%")
        print("=" * 55)


if __name__ == "__main__":
    print("📖 Loading TEST split from Hugging Face Hub...")
    _, _, test_set = ResumeItem.from_hub("alecs-vlad24/cv-resume-structuring")
    evaluator = BaselineEvaluator(
        ollama_baseline_predictor, test_set, size=50, workers=2
    )
    evaluator.run()
