import os
import json
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
from tavily import TavilyClient

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, AIMessage

load_dotenv(override=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file!")

client = OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

LLM_MODEL = "gpt-4o-mini"


class GapAnalysisResult(BaseModel):
    match_percentage: int = Field(description="Match percentage between 0 and 100")
    matching_skills: List[str] = Field(
        description="Skills the candidate already possesses according to requirements"
    )
    missing_skills: List[str] = Field(
        description="Job-required skills missing from candidate's profile"
    )
    transferable_skills: List[str] = Field(
        description="Related skills applicable to the new role"
    )
    seniority_fit: str = Field(
        description="Seniority assessment (e.g., 'Direct Fit', 'Under-qualified', 'Over-qualified')"
    )
    justification: str = Field(
        description="Concise technical explanation justifying the calculated score and seniority fit."
    )


# Performs technical gap analysis comparing the candidate CV with the target job requirements.
def tool_cv_gap_analyzer(cv_summary: str, job_description: str) -> Dict[str, Any]:
    print(
        "\n🔍 [Tool 1: Gap Analyzer] Evaluating technical compatibility between CV and Job..."
    )

    system_prompt = """You are a Principal Technical Recruiter and Engineering Hiring Manager.
Your task is to conduct a strict, objective, and realistic technical Gap Analysis between a Candidate's CV profile and a Target Job Description.

EVALUATION GUIDELINES:
1. Match Percentage (0-100%):
   - 80-100%: Candidate meets almost all core requirements and has matching seniority.
   - 50-79%: Candidate has strong fundamentals/related stack, but lacks 1-2 critical core technologies or seniority depth.
   - 0-49%: Significant technical mismatch or complete transition to a new specialization.
2. Skill Classification:
   - matching_skills: Exact or equivalent technologies possessed by candidate.
   - missing_skills: Core stack items strictly required by the job that candidate does NOT have.
   - transferable_skills: Candidate's existing knowledge that directly helps master the missing skills.
3. Seniority & Experience Alignment:
   - Explicitly take into account the candidate's `years_of_experience` and `experience_level` compared to the target role's expectations.
   - Set seniority_fit accordingly ('Direct Fit', 'Slightly Under-qualified', 'Major Gap', 'Over-qualified').

Return strictly the structured data matching the schema.
"""

    user_content = f"""CANDIDATE CV PROFILE:
{cv_summary}

TARGET JOB DESCRIPTION:
{job_description}
"""

    completion = client.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=GapAnalysisResult,
        temperature=0.0,
    )

    result: GapAnalysisResult = completion.choices[0].message.parsed
    res_dict = result.model_dump()

    print(
        f"   👉 Match Score: {res_dict['match_percentage']}% | Missing Skills: {res_dict['missing_skills']}"
    )
    return res_dict


class RoadmapPhase(BaseModel):
    phase_number: int = Field(description="Phase sequence number (1, 2, 3...)")
    title: str = Field(description="Title of the learning phase")
    duration_weeks: int = Field(description="Estimated duration in weeks")
    focus_technologies: List[str] = Field(
        description="Core technologies covered in this phase"
    )
    recommended_resources: List[str] = Field(
        description="Concrete resources (documentation, courses, books)"
    )
    practical_milestone: str = Field(
        description="Practical validation milestone project"
    )


class RoadmapOutput(BaseModel):
    total_estimated_weeks: int = Field(
        description="Total estimated weeks to complete the full learning roadmap."
    )
    phases: List[RoadmapPhase] = Field(
        description="List of progressive learning phases (2 to 4 phases)."
    )
    learning_strategy: str = Field(
        description="Pedagogical strategy regarding technical sequencing and priorities."
    )


# Generates a phased, weekly learning roadmap to bridge identified technical skill gaps.
def tool_roadmap_generator(
    missing_skills: List[str], target_role: str, experience_level: str
) -> Dict[str, Any]:
    print(
        f"\n🗺️ [Tool 2: Roadmap Generator] Building learning roadmap for {len(missing_skills)} missing skills..."
    )

    if not missing_skills:
        print("   ⚡ No missing skills detected. Returning empty roadmap.")
        return {
            "total_estimated_weeks": 0,
            "phases": [],
            "learning_strategy": "Candidate already possesses all required technical proficiencies.",
        }

    system_prompt = f"""You are a Lead Curriculum Architect and Senior Engineering Mentor.
Your task is to design a realistic, step-by-step technical learning roadmap for a candidate aiming to become a '{target_role}' (target level: {experience_level}).

PEDAGOGICAL DESIGN RULES:
1. Divide the learning path into 2 to 4 logical, progressive phases:
   - Phase 1: Core Fundamentals & Direct Prerequisites
   - Phase 2: Core Tooling & Production Workflows
   - Phase 3/4: Advanced Integration, Architecture & Orchestration
2. Assign realistic durations (e.g., 2-4 weeks per phase depending on complexity).
3. Every single phase MUST have:
   - Concrete tools/technologies in `focus_technologies`.
   - Concrete, high-value resource types in `recommended_resources` (official docs, interactive labs, books).
   - A comprehensive, highly descriptive `practical_milestone`:
       *It MUST explicitly incorporate and combine the tools listed in `focus_technologies`.
       *Structure the project description clearly with:
        -Project Scenario & Architecture (e.g., 'Deploying a resilient 3-tier microservice architecture...')
        -Core Technical Tasks (e.g., 'Write modular Terraform definitions to provision AWS VPC, EKS, and RDS with remote state in S3/DynamoDB...')
        -Production Standards (e.g., zero-downtime rolling updates, secret management, IAM least privilege)
        -Deliverable Proof (e.g., GitHub repository with automated GitHub Actions workflow and Prometheus monitoring dashboard).
       *Avoid brief one-liners; make each milestone read like a real-world enterprise engineering assignment. 

Output strictly according to the schema.
"""

    user_content = f"""MISSING SKILLS TO MASTER:
{missing_skills}

TARGET ROLE & SENIORITY:
Role: {target_role} | Level: {experience_level}
"""

    completion = client.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=RoadmapOutput,
        temperature=0.2,
    )

    result: RoadmapOutput = completion.choices[0].message.parsed
    res_dict = result.model_dump()

    print(
        f"   👉 Generated Roadmap: {len(res_dict['phases'])} phases | Total: {res_dict['total_estimated_weeks']} weeks."
    )
    return res_dict


# Searches the live web via Tavily for verified learning resources and technical documentation.
def tool_web_search(query: str, max_results: int = 3) -> str:
    print(
        f"\n🌐 [Tool 3: Web Search] Searching educational resources for: '{query}'..."
    )

    if not tavily_client:
        print("   ⚠️ Tavily client not configured. Returning fallback context.")
        return f"Standard learning resources and official documentation for '{query}'."

    try:
        search_query = f"{query} tutorial official documentation hands-on labs roadmap"

        response = tavily_client.search(
            query=search_query, max_results=max_results, search_depth="basic"
        )

        results = response.get("results", [])
        if not results:
            return f"No relevant web resources found for query: '{query}'."

        formatted_results = []
        for idx, item in enumerate(results, 1):
            title = item.get("title", "No Title")
            url = item.get("url", "")
            content = item.get("content", "").strip()[:320]
            formatted_results.append(
                f"[{idx}] {title}\n    Link: {url}\n    Description: {content}..."
            )

        summary = "\n\n".join(formatted_results)
        print(f"   ✅ Successfully retrieved {len(results)} verified web resources.")
        return summary

    except Exception as e:
        print(f"   ⚠️ Error during Tavily search: {e}")
        return f"Web search error for '{query}': {str(e)}"


class InterviewQuestion(BaseModel):
    category: str = Field(
        description="Question category (e.g., 'Technical Deep-Dive', 'System Architecture', 'Scenario-Based')"
    )
    question: str = Field(description="Technical interview question")
    ideal_answer_points: List[str] = Field(
        description="Key technical concepts that must be covered in an ideal answer"
    )


class InterviewQuestionsOutput(BaseModel):
    questions: List[InterviewQuestion] = Field(
        description="List of 4-6 categorized technical interview questions."
    )
    preparation_tips: str = Field(
        description="Strategic recommendations for approaching technical interviews."
    )


# Generates targeted technical interview questions calibrated on skill gaps and transitional background.
def tool_interview_questions_generator(
    job_role: str,
    missing_skills: List[str],
    existing_skills: List[str],
    experience_level: str,
) -> Dict[str, Any]:
    print(
        f"\n🎙️ [Tool 4: Targeted Interview Prep] Generating questions for gaps: {missing_skills}..."
    )

    system_prompt = f"""You are a Principal Technical Interviewer and Engineering Hiring Manager.
Your task is to generate realistic, targeted technical interview questions for a candidate preparing for the role of '{job_role}' ({experience_level} level).

STRATEGIC INTERVIEW DESIGN RULES:
1. Primary Focus (60%): Focus heavily on the candidate's IDENTIFIED GAPS ({missing_skills}). Ask practical, production-level questions to test their mastery of these newly acquired skills.
2. Bridge & Integration (40%): Formulate questions that connect the candidate's EXISTING KNOWLEDGE ({existing_skills}) with the MISSING SKILLS (e.g., integrating their known backend stack into newly learned cloud/container infrastructure).
3. Question Quality:
   - NO generic definitions (e.g., avoid 'What is Terraform?').
   - Include realistic failure modes, debugging scenarios, and architecture trade-offs.
   - For every question, provide 3-4 concrete, technical bullet points in `ideal_answer_points`.

Output strictly according to the schema.
"""

    user_content = f"""CANDIDATE EXISTING SKILLS:
{json.dumps(existing_skills)}

SKILL GAPS (CRITICAL FOCUS):
{json.dumps(missing_skills)}

TARGET ROLE & SENIORITY:
Role: {job_role} | Level: {experience_level}
"""

    completion = client.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=InterviewQuestionsOutput,
        temperature=0.2,
    )

    result: InterviewQuestionsOutput = completion.choices[0].message.parsed
    res_dict = result.model_dump()

    print(
        f"   👉 Generated {len(res_dict['questions'])} targeted technical interview questions."
    )
    return res_dict


class PortfolioProject(BaseModel):
    project_title: str = Field(
        description="Technical title of the portfolio project (e.g., 'Cloud-Native Microservices Platform on AWS with GitOps CI/CD')."
    )
    business_context: str = Field(
        description="Business problem solved and production value provided by the system."
    )
    target_role_alignment: str = Field(
        description="Explanation of how this project demonstrates mastery of the missing skills and job requirements."
    )
    tech_stack_used: List[str] = Field(
        description="Complete list of technologies, frameworks, IaC tools, and cloud services used."
    )
    architecture_overview: str = Field(
        description="Description of data flow, component integration, and system architecture."
    )
    key_features_to_implement: List[str] = Field(
        description="List of 4-6 major technical features."
    )
    step_by_step_implementation_guide: List[str] = Field(
        description="Concrete sequential steps from initial repository setup to production deployment."
    )
    github_readme_highlights: str = Field(
        description="Recommendations for architectural diagrams, setup guides, and validation proofs in the README."
    )


class FinalCareerReport(BaseModel):
    target_role: str = Field(description="Analyzed target role title.")
    executive_summary: str = Field(
        description="Strategic 2-3 sentence executive summary assessing the candidate profile and transition feasibility."
    )
    gap_analysis: GapAnalysisResult = Field(
        description="Full technical competence analysis (match %, seniority fit, matching & missing skills, justification)."
    )
    learning_roadmap: List[RoadmapPhase] = Field(
        description="Phased weekly plan for acquiring missing competencies."
    )
    portfolio_project: PortfolioProject = Field(
        description="Comprehensive portfolio project unifying learned and existing technologies."
    )
    interview_preparation: List[InterviewQuestion] = Field(
        description="Categorized technical interview questions with ideal response points."
    )


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "tool_cv_gap_analyzer",
            "description": "Performs a strict technical gap analysis comparing the candidate's CV profile against the target job description. Evaluates technical fit, seniority alignment, matching skills, missing skills, and transferable skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cv_summary": {
                        "type": "string",
                        "description": "The JSON-formatted summary or profile of the candidate's CV received from Agent 1.",
                    },
                    "job_description": {
                        "type": "string",
                        "description": "The target job description and requirements selected from Agent 2.",
                    },
                },
                "required": ["cv_summary", "job_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_roadmap_generator",
            "description": "Generates a structured, phased, and weekly learning roadmap with clear milestones to help the candidate master the identified missing technical skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "missing_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of core technologies or competencies strictly missing from the candidate's profile.",
                    },
                    "target_role": {
                        "type": "string",
                        "description": "The title of the target position (e.g., 'Mid-Level DevOps Engineer').",
                    },
                    "experience_level": {
                        "type": "string",
                        "description": "The target seniority level expected for the role (e.g., 'Junior', 'Mid', 'Senior').",
                    },
                },
                "required": ["missing_skills", "target_role", "experience_level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_web_search",
            "description": "Searches the live web via Tavily for verified, up-to-date learning resources, official documentation, hands-on tutorials, and industry certifications for missing tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The technical search keywords (e.g., 'Terraform AWS hands-on labs tutorial').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "The maximum number of search results to return (default is 3).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_interview_questions_generator",
            "description": "Generates realistic, production-oriented technical interview questions and scenario-based questions targeting the candidate's skill gaps and bridging their existing background.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_role": {
                        "type": "string",
                        "description": "The target job role for the interview.",
                    },
                    "missing_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of missing skills that need deep-dive technical assessment.",
                    },
                    "existing_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of the candidate's current technical skills to build bridge and integration questions.",
                    },
                    "experience_level": {
                        "type": "string",
                        "description": "The target experience level for calibration.",
                    },
                },
                "required": [
                    "job_role",
                    "missing_skills",
                    "existing_skills",
                    "experience_level",
                ],
            },
        },
    },
]


class Agent3State(TypedDict):
    cv_data: Dict[str, Any]
    selected_job: Dict[str, Any]
    messages: Annotated[List[AnyMessage], add_messages]
    loop_step: int
    final_report: Optional[Dict[str, Any]]


# Orchestrates autonomous tool selection and reasoning steps in the ReAct loop.
def agent_reasoner_node(state: Agent3State) -> Dict[str, Any]:
    messages = list(state["messages"])
    loop_step = state.get("loop_step", 0) + 1

    print(f"\n🧠 [Agent 3 Reasoner] ReAct Reasoning Round {loop_step}...")

    system_prompt = """You are an Autonomous Senior Technical Career Strategist and Principal Engineering Mentor.

OPERATING CONSTRAINTS:
1. You are strictly a tool-orchestration agent. You do NOT have the authority or internal capability to evaluate CV matches, design roadmaps, generate interview questions, or retrieve verified resources on your own.
2. Every piece of analysis MUST be delegated to your available specialized tools.
3. You must NOT provide a final text response to the user until you have actively gathered data from your tools.
4. You are completely autonomous in deciding which tools to call, in what order, or in parallel.
5. Once all relevant tool findings have been gathered into the conversation, summarize your findings in a brief final thought to finish.
"""

    formatted_messages = [{"role": "system", "content": system_prompt}]
    has_tool_messages = False

    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            has_tool_messages = True
            formatted_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": str(msg.content),
                }
            )
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            formatted_messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"])
                                if isinstance(tc["args"], dict)
                                else tc["args"],
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
        else:
            formatted_messages.append(
                {"role": "assistant", "content": str(msg.content)}
            )

    tool_choice_mode = "required" if not has_tool_messages else "auto"

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=formatted_messages,
        tools=tools_schema,
        tool_choice=tool_choice_mode,
        temperature=0.1,
    )

    choice = response.choices[0].message

    if choice.tool_calls:
        tool_calls_langchain = [
            {
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
                "id": tc.id,
            }
            for tc in choice.tool_calls
        ]
        agent_msg = AIMessage(
            content=choice.content or "", tool_calls=tool_calls_langchain
        )
        print(
            f"   👉 Autonomous Decision: Invoking tools {[tc['name'] for tc in tool_calls_langchain]}"
        )
    else:
        agent_msg = AIMessage(content=choice.content or "")
        print("   👉 Autonomous Decision: All necessary tool findings collected.")

    return {"messages": [agent_msg], "loop_step": loop_step}


# Executes selected tools with arguments requested by the LLM and formats results as ToolMessages.
def tool_executor_node(state: Agent3State) -> Dict[str, Any]:
    last_message = state["messages"][-1]
    tool_messages: List[ToolMessage] = []

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        print("   ⚠️ [Tool Executor] No tool calls requested.")
        return {"messages": []}

    available_tools = {
        "tool_cv_gap_analyzer": tool_cv_gap_analyzer,
        "tool_roadmap_generator": tool_roadmap_generator,
        "tool_web_search": tool_web_search,
        "tool_interview_questions_generator": tool_interview_questions_generator,
    }

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        call_id = tool_call["id"]

        print(f"\n⚙️ [Tool Executor] Executing tool '{tool_name}' with args:")
        print(f"   Args: {json.dumps(tool_args, ensure_ascii=False)[:180]}...")

        if tool_name in available_tools:
            func = available_tools[tool_name]
            try:
                observation = func(**tool_args)

                if isinstance(observation, (dict, list)):
                    obs_content = json.dumps(observation, ensure_ascii=False)
                else:
                    obs_content = str(observation)

            except Exception as e:
                print(f"   ❌ Execution error in tool '{tool_name}': {e}")
                obs_content = json.dumps({"error": f"Tool execution failed: {str(e)}"})
        else:
            print(f"   ❌ Error: Tool '{tool_name}' is not registered.")
            obs_content = json.dumps(
                {"error": f"Tool '{tool_name}' is not recognized."}
            )

        tool_messages.append(
            ToolMessage(content=obs_content, tool_call_id=call_id, name=tool_name)
        )

    return {"messages": tool_messages}


# Evaluates ReAct loop termination conditions based on tool call requests and safety limits.
def should_continue(state: Agent3State) -> str:
    last_message = state["messages"][-1]
    loop_step = state.get("loop_step", 0)

    if loop_step >= 6:
        print(
            "\n⚠️ [Safety Circuit Breaker] Maximum ReAct steps reached (6). Proceeding to synthesis."
        )
        return "synthesize"

    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "execute_tools"

    print("\n✅ [ReAct Loop Complete] Synthesizing final career transition report...")
    return "synthesize"


# Aggregates all tool observations and candidate data into the final structured career report.
def report_synthesizer_node(state: Agent3State) -> Dict[str, Any]:
    print("\n📝 [Report Synthesizer] Compiling final structured career report...")

    cv_data = state["cv_data"]
    selected_job = state["selected_job"]
    messages = state["messages"]

    tool_observations = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_observations.append(
                f"--- OBSERVATION FROM TOOL [{msg.name}] ---\n{msg.content}"
            )

    accumulated_context = (
        "\n\n".join(tool_observations)
        if tool_observations
        else "No explicit tool observations."
    )

    system_prompt = """You are a Principal Engineering Architect, Lead Career Mentor, and Technical Hiring Strategist.
Your mission is to synthesize all accumulated tool observations and profile inputs into an elite, end-to-end Final Career Report.

SYNTHESIS GUIDELINES:
1. Executive Summary: Write a sharp, 2-3 sentence strategic executive summary assessing the candidate's current baseline, transition feasibility, and core priority.
2. Gap Analysis & Roadmap: Preserve the rigorous technical gap analysis and progressive learning roadmap gathered from the tools.
3. Portfolio Project: Design a tangible, production-grade portfolio project that combines the candidate's existing strengths with all the newly learned missing technologies. Ensure architecture details, key features, step-by-step guides, and GitHub README requirements are extensive and descriptive.
4. Interview Preparation: Ensure technical interview questions target the critical gaps and transition bridge points.

Output strictly adhering to the FinalCareerReport schema.
"""

    user_content = f"""CANDIDATE INITIAL CV PROFILE:
{json.dumps(cv_data, indent=2, ensure_ascii=False)}

TARGET JOB REQUIREMENTS:
{json.dumps(selected_job, indent=2, ensure_ascii=False)}

ALL ACCUMULATED TOOL FINDINGS & TECHNICAL OBSERVATIONS:
{accumulated_context}
"""

    completion = client.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=FinalCareerReport,
        temperature=0.2,
    )

    final_report_obj: FinalCareerReport = completion.choices[0].message.parsed
    final_report_dict = final_report_obj.model_dump()

    print("🎉 [Agent 3 Complete] Strategic career report compiled successfully!")
    return {"final_report": final_report_dict}


# Constructs and compiles the LangGraph state graph for Agent 3.
def build_agent3_graph():
    builder = StateGraph(Agent3State)

    builder.add_node("reasoner", agent_reasoner_node)
    builder.add_node("executor", tool_executor_node)
    builder.add_node("synthesizer", report_synthesizer_node)

    builder.set_entry_point("reasoner")

    builder.add_conditional_edges(
        "reasoner",
        should_continue,
        {"execute_tools": "executor", "synthesize": "synthesizer"},
    )

    builder.add_edge("executor", "reasoner")
    builder.add_edge("synthesizer", END)

    return builder.compile()


agent3_graph = build_agent3_graph()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 END-TO-END UNIT TEST: AUTONOMOUS ReAct LOOP")
    print("=" * 70)

    mock_cv = {
        "candidate_role": "Python Developer",
        "experience_level": "Junior",
        "years_of_experience": 1.5,
        "primary_skills": ["Python", "FastAPI", "Docker", "PostgreSQL", "Git"],
        "clean_summary": "Junior Python Developer with 1.5 years experience building backend APIs and working with relational databases.",
    }

    mock_job = {
        "title": "Mid-Level DevOps Engineer",
        "experience_level": "Mid",
        "primary_tech_stack": [
            "AWS",
            "Terraform",
            "Docker",
            "Kubernetes",
            "CI/CD",
            "Linux",
        ],
        "responsibilities": [
            "Build and manage cloud infrastructure using Terraform on AWS",
            "Deploy and maintain Kubernetes clusters in production",
            "Maintain CI/CD pipelines for automated testing and deployment",
        ],
    }

    initial_state: Agent3State = {
        "cv_data": mock_cv,
        "selected_job": mock_job,
        "messages": [
            HumanMessage(
                content="Please analyze the CV for the target job, construct a roadmap, search resources, and prepare interview questions to generate the final career report."
            )
        ],
        "loop_step": 0,
        "final_report": None,
    }

    print("\n🔄 Starting autonomous LangGraph execution...")
    final_output = agent3_graph.invoke(initial_state)
    report = final_output["final_report"]

    print("\n" + "=" * 70)
    print("🏆 FINAL CAREER REPORT GENERATED BY AGENT 3:")
    print("=" * 70)
    print(f"\n🎯 Target Role: {report['target_role']}")
    print(f"\n📌 Executive Summary:\n{report['executive_summary']}")
    print(
        f"\n📊 Gap Analysis: Match {report['gap_analysis']['match_percentage']}% | Seniority Fit: {report['gap_analysis']['seniority_fit']}"
    )
    print(f"   Missing Skills: {report['gap_analysis']['missing_skills']}")
    print(f"\n🛣️ Roadmap: {len(report['learning_roadmap'])} phases generated.")
    print(f"\n🏗️ Portfolio Project: {report['portfolio_project']['project_title']}")
    print(f"   Business Context: {report['portfolio_project']['business_context']}")
    print(
        f"\n🎙️ Interview Questions: {len(report['interview_preparation'])} technical questions prepared."
    )
