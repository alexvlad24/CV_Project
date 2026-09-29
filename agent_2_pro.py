import os
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, FusionQuery, Fusion, Prefetch
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder
import json
from pydantic import BaseModel, Field
from tavily import TavilyClient

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "it_jobs"
EMBEDDING_MODEL = "text-embedding-3-small"

qdrant_client = QdrantClient(path=QDRANT_PATH)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

load_dotenv(override=True)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file!")

client = OpenAI(api_key=OPENAI_API_KEY)


class Agent2State(TypedDict):
    cv_text: str
    user_target_text: str
    resolved_role: str
    hyde_document: str
    candidate_jobs: List[Dict[str, Any]]
    top_jobs: List[Dict[str, Any]]
    best_score: float
    retry_count: int
    feedback: Optional[str]
    source: str


class RoleExtractionResponse(BaseModel):
    is_specific_request: bool = Field(
        description="True if user specified a clear role, tech stack, or career direction. False if vague, generic, or asking for open recommendations."
    )
    extracted_role: Optional[str] = Field(
        default=None,
        description="The standardized IT role and seniority synthesized from the user input (e.g., 'Junior DevOps Engineer', 'Mid-Level React Developer'). Null if is_specific_request is False.",
    )


# Extracts and standardizes the target IT role from user input or falls back to the candidate's CV profile.
def resolve_role_node(state: Agent2State) -> Dict[str, Any]:
    user_target = state.get("user_target_text", "").strip()
    cv_data_raw = state.get("cv_text", "")

    cv_json = {}
    if isinstance(cv_data_raw, str) and cv_data_raw.strip():
        try:
            cv_json = json.loads(cv_data_raw)
        except Exception:
            cv_json = {}
    elif isinstance(cv_data_raw, dict):
        cv_json = cv_data_raw

    extracted_role = cv_json.get("candidate_role") or cv_json.get("Candidate_Role", "")
    extracted_level = cv_json.get("experience_level") or cv_json.get(
        "Experience_Level", ""
    )

    if extracted_role:
        if extracted_level and extracted_level.lower() not in extracted_role.lower():
            fallback_cv_role = f"{extracted_level} {extracted_role}".strip()
        else:
            fallback_cv_role = extracted_role.strip()
    else:
        fallback_cv_role = "Software Engineer"

    print("\n🎯 [Node 1: Resolve Role] Processing request...")

    if not user_target:
        print(
            f"   ⚡ Empty input field -> Using CV role directly: '{fallback_cv_role}'"
        )
        return {"resolved_role": fallback_cv_role, "retry_count": 0, "feedback": None}

    system_prompt = """You are an expert IT Talent Profiler and Technical Recruiter.
Analyze the user's input request:

1. If the User Input contains a specific job title, tech stack preference, or career description:
   - Synthesize and extract the most accurate IT Role Title with its seniority level if detectable (e.g., "Junior DevOps Engineer", "Mid-Level React Developer", "Senior Cloud Architect") based on user input
   - Populate `extracted_role` with this role
   - Set is_specific_request = True
2. If the input is vague, open-ended, generic, or asks for recommendations (e.g., 'what fits me?', 'recommend me something', 'anything in tech'):
   - Set is_specific_request = False
   - Set extracted_role = null

RULES:
1. Output STRICTLY the role title with its seniority level if detectable (e.g., "Mid-Level DevOps Engineer"). 
2. Do NOT output any markdown, explanations, punctuation, or extra words. Output ONLY the role name.
"""

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'User Input: "{user_target}"'},
        ],
        response_format=RoleExtractionResponse,
        temperature=0.0,
    )

    parsed_result: RoleExtractionResponse = completion.choices[0].message.parsed

    if parsed_result.is_specific_request and parsed_result.extracted_role:
        resolved_role = parsed_result.extracted_role.strip()
        print(f"   👉 Role extracted from user preferences: '{resolved_role}'")
    else:
        resolved_role = fallback_cv_role
        print(
            f"   🔄 Generic input detected -> Falling back to CV role: '{resolved_role}'"
        )

    return {"resolved_role": resolved_role, "retry_count": 0, "feedback": None}


# Generates a synthetic, detailed job description (HyDE) tailored to the resolved target role and optional feedback.
def hyde_generator_node(state: Agent2State) -> Dict[str, Any]:
    target_role = state.get("resolved_role", "Software Engineer")
    feedback = state.get("feedback")

    print(
        f"\n🧠 [Node 2: HyDE Generator] Generating hypothetical job profile for: '{target_role}'..."
    )
    if feedback:
        print(f"   ⚙️ Applying diagnostic feedback: {feedback}")

    feedback_instruction = (
        f"\nCRITICAL ADJUSTMENT FEEDBACK FROM PREVIOUS SEARCH ATTEMPT:\n{feedback}\n"
        if feedback
        else ""
    )

    system_prompt = f"""You are an expert IT Talent Acquisition Specialist and Lead Systems Architect.
Your task is to generate a realistic, detailed, and highly technical job description draft for a given IT role.

Focus strictly on:
1. Primary Tech Stack (concrete tools, frameworks, programming languages, databases, cloud providers).
2. Required Skills & Engineering Methodologies.
3. Core Technical Responsibilities (concise, high-impact technical duty statements starting with strong action verbs; focus strictly on system architecture, code delivery, and infrastructure management).

CRITICAL RULES:
- Keep it 100% focused on technical requirements and core duties.
- Do NOT include generic non-technical HR filler text (salaries, perks, company culture, soft skills).
- Avoid vague terms like 'scripting skills' or 'cloud concepts'; always specify concrete tools or framework names (e.g., 'Docker', 'Kubernetes', 'PostgreSQL', 'AWS').
- Always include modern industry-standard technologies.
- Match technologies strictly to the specific role.
- DYNAMIC SENIORITY ADAPTATION: If the role name explicitly contains a seniority level (e.g., 'Junior', 'Intern', 'Mid', 'Senior', 'Lead'), adapt the technology scope, responsibility complexity, and skill depth specifically for that level.
{feedback_instruction}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{target_role}"},
        ],
        temperature=0.2,
    )

    hyde_document = response.choices[0].message.content.strip()
    return {"hyde_document": hyde_document}


# Performs hybrid search (Dense embeddings + Sparse BM25) with Reciprocal Rank Fusion against local Qdrant collection.
def hybrid_search_node(state: Agent2State) -> Dict[str, Any]:
    expanded_query = state.get("hyde_document", "").strip()
    top_k = 12

    print(
        f"\n🔍 [Node 3: Hybrid Search] Searching top {top_k} candidate jobs in Qdrant..."
    )

    if not expanded_query:
        print("   ⚠️ No HyDE document found in state. Returning empty list.")
        return {"candidate_jobs": []}

    dense_resp = client.embeddings.create(model=EMBEDDING_MODEL, input=expanded_query)
    dense_vector = dense_resp.data[0].embedding

    sparse_emb = list(sparse_model.embed([expanded_query]))[0]
    sparse_vector = SparseVector(
        indices=sparse_emb.indices.tolist(), values=sparse_emb.values.tolist()
    )

    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(using="dense", query=dense_vector, limit=top_k * 2),
            Prefetch(using="sparse", query=sparse_vector, limit=top_k * 2),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    )

    retrieved_jobs = []
    for point in search_results.points:
        job_data = dict(point.payload) if point.payload else {}
        job_data["_score"] = round(point.score, 4) if point.score else 0.0
        retrieved_jobs.append(job_data)

    print(
        f"   ✅ [Node 3 Qdrant] Retrieved {len(retrieved_jobs)} candidate job profiles."
    )

    return {"candidate_jobs": retrieved_jobs}


# Evaluates candidate profiles using a local cross-encoder model and selects the top 3 best matching job descriptions.
def reranker_node(state: Agent2State) -> Dict[str, Any]:
    candidate_jobs = state.get("candidate_jobs", [])
    query_doc = state.get("hyde_document", "").strip()
    top_n = 3

    print(
        f"\n🎯 [Node 4: Reranker] Re-ranking {len(candidate_jobs)} candidate job profiles..."
    )

    if not candidate_jobs or not query_doc:
        print("   ⚠️ Missing candidate jobs or HyDE document for reranking.")
        return {"top_jobs": [], "best_score": -99.0, "source": "qdrant"}

    pairs = []
    for job in candidate_jobs:
        job_text = (
            f"Title: {job.get('title')} | Level: {job.get('experience_level')} | "
            f"Tech Stack: {', '.join(job.get('primary_tech_stack', []))} | "
            f"Responsibilities: {'; '.join(job.get('responsibilities', []))}"
        )
        pairs.append((query_doc, job_text))

    scores = reranker_model.predict(pairs)

    for idx, job in enumerate(candidate_jobs):
        job["rerank_score"] = round(float(scores[idx]), 4)

    sorted_jobs = sorted(candidate_jobs, key=lambda x: x["rerank_score"], reverse=True)

    top_jobs = sorted_jobs[:top_n]
    best_score = top_jobs[0]["rerank_score"] if top_jobs else -99.0

    print(f"   👉 Top 1 Reranker Score: {best_score}")
    print(f"   ✅ Selected Top {len(top_jobs)} re-ranked job profiles.")

    return {"top_jobs": top_jobs, "best_score": best_score, "source": "qdrant"}


# Analyzes search failure causes when relevance score is below threshold and produces feedback to improve HyDE generation.
def diagnose_and_reformulate_node(state: Agent2State) -> Dict[str, Any]:
    resolved_role = state.get("resolved_role", "")
    current_hyde = state.get("hyde_document", "")
    best_score = state.get("best_score", -99.0)
    current_retry = state.get("retry_count", 0)

    print(
        f"\n🧐 [Node 5: Diagnosis] Reranker score below threshold ({best_score:.4f}). Analyzing cause..."
    )

    diagnostic_prompt = f"""You are an expert Search Relevance and IT Query Diagnostics Specialist.
We are searching a vector database of IT job descriptions for the target role: '{resolved_role}'.
The previous hypothetical job description (HyDE) generated for this role yielded poor match scores.

Generated HyDE (Excerpt):
"{current_hyde[:600]}"

Your Task:
Diagnose why this generated job description might not match standard real-world job profiles in our database and provide 1-2 concise, actionable instructions to adjust the next HyDE generation.

Failure patterns to consider:
1. Over-specialization: Required too many niche/rare tools simultaneously -> Instruct to focus on core industry standards.
2. Too Abstract: Lacked concrete tools, libraries, or frameworks -> Instruct to add explicit, widely used tech stack items.
3. Seniority Imbalance: Scope of technical duties didn't align with the role title -> Instruct to recalibrate technical responsibilities.

Rules:
- Return ONLY the actionable correction instruction (1-2 sentences).
- Do not output markdown, pleasantries, or preamble.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": diagnostic_prompt}],
        temperature=0.0,
    )

    feedback = response.choices[0].message.content.strip().replace('"', "")
    print(f'   👉 Formulated diagnostic feedback: "{feedback}"')

    return {"retry_count": current_retry + 1, "feedback": feedback}


class JobProfileSchema(BaseModel):
    title: str = Field(description="Standardized technical job title")
    experience_level: str = Field(description="Junior, Mid, or Senior")
    primary_tech_stack: List[str] = Field(
        description="List of core tools, libraries, languages, and frameworks"
    )
    responsibilities: List[str] = Field(
        description="List of 3-5 technical engineering duties starting with action verbs"
    )
    rerank_score: float = Field(default=0.0)


class WebSearchJobExtraction(BaseModel):
    jobs: List[JobProfileSchema] = Field(
        description="Exactly 3 distinct structured job descriptions"
    )


TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


# Executes web search retrieval via Tavily and extracts 3 structured job descriptions when local search fails.
def web_search_fallback_node(state: Agent2State) -> Dict[str, Any]:
    target_role = state.get("resolved_role", "Software Engineer")
    print(
        f"\n🌐 [Node Fallback: Web Search] Role '{target_role}' not sufficiently matched locally. Searching Web..."
    )

    search_query = f"technical job description requirements responsibilities tech stack {target_role}"
    web_context = ""

    if tavily_client:
        try:
            res = tavily_client.search(query=search_query, max_results=3)
            web_context = "\n".join(
                [
                    f"- {r.get('title')}: {r.get('content')}"
                    for r in res.get("results", [])
                ]
            )
            print("   📡 Retrieved live web results via Tavily.")
        except Exception as e:
            print(
                f"   ⚠️ Tavily unavailable ({e}). Proceeding with autonomous synthesis."
            )

    system_prompt = f"""You are an expert Technical Job Architect.
Generate exactly 3 realistic, distinct, and complete job profiles for the target role: '{target_role}'.

HOW TO DIFFERENTIATE THE 3 PROFILES:
1. If '{target_role}' DOES NOT specify a seniority level:
   - Profile 1: Junior level (foundational tools, guided development, standard stack)
   - Profile 2: Mid level (independent feature delivery, production microservices, CI/CD)
   - Profile 3: Senior level (system design, scalability, architecture, performance tuning)

2. If '{target_role}' ALREADY specifies a seniority level (e.g., 'Junior', 'Mid', 'Senior'):
   - Keep the specified experience level constant across all 3 profiles.
   - Profile 1: Core/Standard Market Stack (most common industry toolset)
   - Profile 2: Modern/Cloud-Native Stack (modern frameworks, cloud ecosystem)
   - Profile 3: Enterprise/Data-Heavy Variant (robust enterprise tooling or specialized integrations)

REAL-WORLD WEB CONTEXT:
{web_context if web_context else f"Standard industry market requirements for {target_role}."}

CRITICAL SCHEMA RULES:
- Every profile must have an experience_level ('Junior', 'Mid', or 'Senior').
- Every profile must have 4-7 concrete tools/frameworks in primary_tech_stack.
- Every profile must have 3-5 technical engineering duties in responsibilities starting with strong action verbs.
"""

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Synthesize 3 complete job profiles for '{target_role}'.",
            },
        ],
        response_format=WebSearchJobExtraction,
        temperature=0.2,
    )

    parsed_data: WebSearchJobExtraction = completion.choices[0].message.parsed
    top_3_jobs = [job.model_dump() for job in parsed_data.jobs][:3]

    print(
        f"   ✅ Successfully structured {len(top_3_jobs)} complete job profiles from Web."
    )

    return {"top_jobs": top_3_jobs, "best_score": 0.0, "source": "web"}


RELEVANCE_THRESHOLD = -5.3


# Routes execution based on the reranker relevance score and current retry attempt count.
def route_after_rerank(state: Agent2State) -> str:
    best_score = state.get("best_score", -99.0)
    retry_count = state.get("retry_count", 0)

    if best_score >= RELEVANCE_THRESHOLD:
        print("🔀 [Router] Score above threshold -> Finalizing with local results.")
        return "end"
    elif retry_count == 0:
        print(
            "🔀 [Router] Score below threshold (first attempt) -> Routing to Diagnosis & Reformulation."
        )
        return "retry_diagnosis"
    else:
        print("🔀 [Router] Persistent low score -> Routing to Web Search Fallback.")
        return "web_fallback"


workflow = StateGraph(Agent2State)

workflow.add_node("resolve_role", resolve_role_node)
workflow.add_node("hyde_generator", hyde_generator_node)
workflow.add_node("hybrid_search", hybrid_search_node)
workflow.add_node("reranker", reranker_node)
workflow.add_node("diagnose_and_reformulate", diagnose_and_reformulate_node)
workflow.add_node("web_search_fallback", web_search_fallback_node)

workflow.set_entry_point("resolve_role")
workflow.add_edge("resolve_role", "hyde_generator")
workflow.add_edge("hyde_generator", "hybrid_search")
workflow.add_edge("hybrid_search", "reranker")

workflow.add_conditional_edges(
    "reranker",
    route_after_rerank,
    {
        "end": END,
        "retry_diagnosis": "diagnose_and_reformulate",
        "web_fallback": "web_search_fallback",
    },
)

workflow.add_edge("diagnose_and_reformulate", "hyde_generator")
workflow.add_edge("web_search_fallback", END)

agent2_app = workflow.compile()


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("🧪 UNIT TEST: NODE 1 (RESOLVE ROLE) - ALL CASES")
    print("=" * 65)

    mock_agent1_cv = json.dumps(
        {
            "candidate_role": "Python Developer",
            "experience_level": "Junior",
            "primary_skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
            "clean_summary": "Junior Python Developer with 1.5 years experience in building APIs.",
        }
    )

    print("\n--- Test 1: Specific Input ---")
    state_1: Agent2State = {
        "cv_text": mock_agent1_cv,
        "user_target_text": "I want to work with AWS, manage Kubernetes clusters and write Terraform scripts.",
        "resolved_role": "",
        "hyde_document": "",
        "candidate_jobs": [],
        "top_jobs": [],
        "best_score": 0.0,
        "retry_count": 0,
        "feedback": None,
        "source": "",
    }
    res_1 = resolve_role_node(state_1)
    print(f"👉 Result 1: {res_1['resolved_role']}")

    print("\n--- Test 2: Generic Input ---")
    state_2: Agent2State = {
        "cv_text": mock_agent1_cv,
        "user_target_text": "I dont know extactly, recommend me something",
        "resolved_role": "",
        "hyde_document": "",
        "candidate_jobs": [],
        "top_jobs": [],
        "best_score": 0.0,
        "retry_count": 0,
        "feedback": None,
        "source": "",
    }
    res_2 = resolve_role_node(state_2)
    print(f"👉 Result 2: {res_2['resolved_role']}")

    print("\n--- Test 3: Empty Input ---")
    state_3: Agent2State = {
        "cv_text": mock_agent1_cv,
        "user_target_text": "",
        "resolved_role": "",
        "hyde_document": "",
        "candidate_jobs": [],
        "top_jobs": [],
        "best_score": 0.0,
        "retry_count": 0,
        "feedback": None,
        "source": "",
    }
    res_3 = resolve_role_node(state_3)
    print(f"👉 Result 3: {res_3['resolved_role']}")
