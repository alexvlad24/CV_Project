import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
from pathlib import Path
from pypdf import PdfReader
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

HF_MODEL_ID = "alecs-vlad24/cv-structuring-llm-v2-2026-08-12_10.49"
BASE_MODEL_ID = "unsloth/llama-3.2-3b-instruct-bnb-4bit"

SYSTEM_PROMPT = """You are an expert HR Data Engineer and Technical Parser. Your sole task is to analyze candidate resumes and extract structured technical information into a strict JSON format.

### GENERAL RULES:
1. IGNORE ALL PII: Completely omit names, phone numbers, email addresses, physical addresses, and links.
2. OUTPUT FORMAT: Respond ONLY with a valid JSON object. Do NOT include any intro, outro, explanations, or conversational filler.
3. FACTUAL EXTRACTION: Rely ONLY on the information present in the text. Do not fabricate experience, companies, or degrees.

### STRICT SKILL EXTRACTION RULES (IMPORTANT):
- EXTRACT ONLY SPECIFIC HARD SKILLS: Extract concrete technologies, programming languages, frameworks, libraries, databases, DevOps tools, and platforms (e.g., "Python", "React", "Docker", "PostgreSQL", "AWS", "Git").
- NO GENERAL/BASIC FILLER TEXT: Absolutely DO NOT include generic phrases or broad terms like "basic web development", "programming", "software engineering", "problem solving", "teamwork", "data analysis", or "database management".
- STANDARDIZE NAMES: Keep skill names short and standardized (e.g., use "React" instead of "react.js" or "React framework").
- COMPREHENSIVE TECH EXTRACTION: Include every explicit technology, framework, database, cloud provider, and tool mentioned across all work history and project sections. Do not truncate the list.

### FIELD DEFINITIONS & EXTRACTION RULES:
- "Candidate_Role": The candidate's primary job title or professional role.
  * If the candidate is a student or entry-level applicant without a formal job title, infer the role based on their primary technical skills and projects (e.g., "Junior Software Engineer", "Data Science Intern").
  * Fallback Rule: Only if the role is completely ambiguous and cannot be inferred from projects/skills, set the role based on their highest field of study or diploma (e.g., "Computer Science Graduate", "Information Systems Student").
    
- "Experience_Level": Overall career seniority.
  * Must be EXACTLY one of: "Entry", "Mid", "Senior", "Lead".
  * Assign "Entry" for students, recent graduates, or candidates with 0-2 years of experience.

- "Years_Of_Experience": An integer representing total estimated years of professional experience.
  * Calculate based on work history dates.
  * For students or candidates with no formal work history, set this to 0.

- "Primary_Skills": An array of strings containing ONLY concrete technical skills and tools, adhering to the STRICT SKILL EXTRACTION RULES above.

- "Clean_Summary": A concise, objective, 2-sentence summary of the candidate's technical profile, key experience, and primary capabilities. Avoid first-person pronouns ("I", "my").

### EXPECTED JSON STRUCTURE:
{
  "Candidate_Role": "string",
  "Experience_Level": "Entry | Mid | Senior | Lead",
  "Years_Of_Experience": integer,
  "Primary_Skills": ["string"],
  "Clean_Summary": "string"
}"""


class Agent1CVExtractor:
    # Initializes the tokenizer and loads the 4-bit quantized base model with the LoRA adapter onto the GPU.
    def __init__(self, model_id: str = HF_MODEL_ID):
        print(f"🔄 Loading fine-tuned model on GPU: '{model_id}'...")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=quant_config,
            device_map={"": 0} if torch.cuda.is_available() else "auto",
        )

        self.model = PeftModel.from_pretrained(base_model, model_id)
        print("✅ Agent 1 model loaded successfully on GPU!")

    # Reads a PDF file from the given path and extracts its textual content.
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"❌ PDF file not found at: {pdf_path}")

        reader = PdfReader(path)
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"

        clean_text = extracted_text.strip()
        if not clean_text:
            raise ValueError("⚠️ The PDF file contains no readable text.")

        return clean_text

    # Strips markdown code blocks and isolates the raw JSON string from the model response.
    def clean_json_output(self, raw_text: str) -> str:
        raw_text = raw_text.strip()
        if "```" in raw_text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if match:
                return match.group(1)
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            return raw_text[start_idx : end_idx + 1]

        return raw_text

    # Processes the input CV (raw text or PDF file) and returns the structured technical profile as a dictionary.
    def run(self, cv_input: str, is_pdf: bool = False) -> dict:
        if is_pdf or (isinstance(cv_input, str) and cv_input.lower().endswith(".pdf")):
            print(f"📄 Extracting text from PDF: '{cv_input}'...")
            raw_cv_text = self.extract_text_from_pdf(cv_input)
        else:
            print("📝 Processing raw text...")
            raw_cv_text = cv_input.strip()

        print("🧠 [Agent 1] Running GPU inference...")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{raw_cv_text}"},
        ]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = model_inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][prompt_length:]
        raw_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        cleaned_json_str = self.clean_json_output(raw_response)

        try:
            parsed_json = json.loads(cleaned_json_str)
            print("✅ [Agent 1] Structured JSON successfully extracted!")
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decoding error: {e}")
            return {
                "Candidate_Role": "Unknown",
                "Experience_Level": "Entry",
                "Years_Of_Experience": 0,
                "Primary_Skills": [],
                "Clean_Summary": raw_cv_text[:200],
            }


if __name__ == "__main__":
    agent_1 = Agent1CVExtractor()

    real_cv_test = """
    JOHN DOE

Full Stack Developer

Professional Summary

Experienced Senior Software Engineer with a strong background in building scalable web applications, microservices, and modern cloud architectures. Skilled across the full development lifecycle, from front-end user interface design to back-end database optimization and automated CI/CD deployment pipelines. Proven track record of improving application performance and maintaining enterprise-level software systems in collaborative, Agile environments.

Work Experience

Senior Software Engineer at Tech Corp (2020 – Present)

Responsible for architecting and developing robust microservices using Python, FastAPI, and PostgreSQL. Spearheaded the construction of responsive, modern front-end user interface components utilizing React, TypeScript, and Tailwind CSS. Managed end-to-end cloud infrastructure and continuous integration workflows by configuring automated CI/CD pipelines with Docker, GitHub Actions, and AWS services including EC2 and S3. Significantly enhanced system performance by optimizing database queries and implementing Redis caching, achieving a 40% reduction in query response times.

Software Developer at Web Solutions (2018 – 2020)

Maintained and updated legacy Java Spring Boot applications alongside core REST APIs to ensure high system availability and seamless data flow. Collaborated daily within an Agile/Scrum cross-functional team, managing and querying both relational and NoSQL databases, including MongoDB and MySQL, to support ongoing product feature releases.

Education

Bachelor of Science in Computer Science

University of Technology (2014 – 2018)

Technical Skills

Programming Languages: Python, JavaScript, TypeScript, Java, SQL

Frameworks & Libraries: FastAPI, React, Spring Boot, Node.js

Databases & Tools: PostgreSQL, Redis, MongoDB, Docker, Git, AWS
    """

    print("\n" + "=" * 80)
    print("🚀 TEST 2: COMPLEX CV")
    print("=" * 80)
    result_json = agent_1.run(real_cv_test, is_pdf=False)
    print(json.dumps(result_json, indent=2, ensure_ascii=False))
