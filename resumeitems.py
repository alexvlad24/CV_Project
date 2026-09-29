from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Self
from datasets import Dataset, DatasetDict, load_dataset
import json

SYSTEM_PROMPT = """You are an expert HR Data Engineer and Technical Resume Parser. Your sole task is to analyze candidate resumes and extract complete, precise, and highly structured technical information into a strict JSON object.

CURRENT TIME CONTEXT:
- The current year is 2026. 
- Any work experience listed as "Present", "Current", "Ongoing", or "Till Date" automatically ends in 2026.

### GENERAL RULES:
1. IGNORE ALL PII: Completely omit candidate names, phone numbers, email addresses, physical addresses, websites, social links, and GitHub/LinkedIn usernames.
2. STRICT JSON OUTPUT: Respond ONLY with a valid, parsable JSON object. Do NOT include markdown wrappers outside the JSON, intro/outro text, or explanations.
3. FACTUAL EXTRACTION ONLY: Extract information strictly present in or directly inferable from the text. Do not invent experience, degrees, or tools.

### FIELD EXTRACTION RULES & CALCULATIONS:

1. Candidate_Role:
   - Identify the candidate's primary target role or actual professional title based on their recent experience and total skill depth (e.g., "Senior Full Stack Engineer", "DevOps Engineer", "Data Scientist").
   - If the candidate is a student/graduate with no explicit title, infer it from their primary projects and studies (e.g., "Junior Software Developer").

2. Years_Of_Experience:
   - Must be an INTEGER representing total professional work experience in years.
   - FORMULA: Calculate the exact duration in years for EACH individual employment entry and SUM them up.
   - RULE FOR "PRESENT": Treat "Present" / "Current" as 2026. (Example: "2020 - Present" = 2026 - 2020 = 6 years. If combined with a previous job "2018 - 2020" = 2 years, Total = 8 years).
   - Round to the nearest whole integer.
   - For students, interns, or candidates with no formal job history, set to 0.

3. Experience_Level:
   - Assign seniority level STRICTLY correlated with calculated "Years_Of_Experience":
     * "Entry": 0 - 2 years
     * "Mid": 3 - 5 years
     * "Senior": 6 - 9 years
     * "Lead": 10+ years OR clear explicit management / Tech Lead responsibilities.

4. Primary_Skills (EXHAUSTIVE TECHNICAL EXTRACTION):
   - NO REASONING OR TOP-N LIMITS: Extract EVERY SINGLE concrete technical tool, programming language, framework, library, database, cloud provider, CI/CD tool, OS, and platform explicitly mentioned anywhere in the resume (Work History, Projects, and Skills section).
   - CONCRETE HARD SKILLS ONLY: Extract concrete technologies (e.g., "Python", "TypeScript", "React", "FastAPI", "Spring Boot", "PostgreSQL", "Redis", "Docker", "AWS", "Git").
   - NO FILLER OR SOFT SKILLS: Absolutely EXCLUDE generic, broad phrases such as "programming", "software development", "web development", "problem solving", "teamwork", "agile", "scrum", "communication", "management", "database management".
   - STANDARDIZATION: Keep names short and canonical (e.g., "React" instead of "React.js", "AWS" instead of "Amazon Web Services").

6. Clean_Summary:
   - A concise, objective, 2-sentence summary highlighting the candidate's primary tech stack, years of experience, and main domain capabilities.
   - Write strictly in the third person. Do NOT use first-person pronouns ("I", "my", "me").

### EXPECTED JSON STRUCTURE:
{
  "Candidate_Role": "string",
  "Experience_Level": "Entry | Mid | Senior | Lead",
  "Years_Of_Experience": integer,
  "Primary_Skills": ["string"],
  "Clean_Summary": "string"
}"""


class ResumeItem(BaseModel):
    category: str
    text: str
    candidate_role: Optional[str] = None
    experience_level: Optional[str] = None
    years_of_experience: Optional[int] = None
    primary_skills: Optional[List[str]] = Field(default_factory=list)
    clean_summary: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None

    # Builds the standard multi-turn conversation messages structure for supervised fine-tuning (SFT).
    def build_sft_messages(self) -> List[Dict[str, str]]:
        assistant_json = {
            "Candidate_Role": self.candidate_role or self.category,
            "Experience_Level": self.experience_level or "Unknown",
            "Years_Of_Experience": self.years_of_experience or 0,
            "Primary_Skills": self.primary_skills,
            "Clean_Summary": self.clean_summary or "",
        }

        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{self.text}"},
            {
                "role": "assistant",
                "content": json.dumps(assistant_json, ensure_ascii=False),
            },
        ]
        return self.messages

    # Returns the string representation of a ResumeItem instance.
    def __repr__(self) -> str:
        return f"<ResumeItem category='{self.category}' length={len(self.text)} chars>"

    # Uploads train, validation, and test dataset splits directly to the Hugging Face Hub.
    @staticmethod
    def push_to_hub(
        dataset_name: str, train: List[Self], val: List[Self], test: List[Self]
    ):
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    # Loads and reconstructs ResumeItem instances from Hugging Face Hub dataset splits.
    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[List[Self], List[Self], List[Self]]:
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
