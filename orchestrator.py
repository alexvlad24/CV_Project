import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from langchain_core.messages import HumanMessage

from agent_1 import Agent1CVExtractor
from agent_2_pro import agent2_app, Agent2State
from agent3 import agent3_graph, Agent3State

print("🚀 [Orchestrator] Initializing Multi-Agent System...")

agent1_extractor = Agent1CVExtractor()


# Dispatches the input CV (PDF or raw text) to Agent 1 for structured technical profile extraction.
def step_1_extract_cv(cv_input: str, is_pdf: bool = True) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("📋 [STEP 1] CV EXTRACTION & STRUCTURING (AGENT 1)")
    print("=" * 70)

    extracted_cv_json = agent1_extractor.run(cv_input, is_pdf=is_pdf)

    role = extracted_cv_json.get("Candidate_Role") or extracted_cv_json.get(
        "candidate_role", "Unknown Role"
    )
    level = extracted_cv_json.get("Experience_Level") or extracted_cv_json.get(
        "experience_level", "Unknown Level"
    )
    skills = extracted_cv_json.get("Primary_Skills") or extracted_cv_json.get(
        "primary_skills", []
    )

    print(f"\n✅ CV successfully processed:")
    print(f"   • Identified Role: {role}")
    print(f"   • Seniority Level: {level}")
    print(f"   • Technical Skills Count: {len(skills)} -> {skills[:6]}...")

    return extracted_cv_json


# Queries Agent 2 using the candidate profile and optional target preference to retrieve the top 3 matching job profiles.
def step_2_search_jobs(
    cv_json: Dict[str, Any], user_target_text: str = ""
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 70)
    print("🔍 [STEP 2] TARGET JOB SEARCH & MATCHING (AGENT 2)")
    print("=" * 70)

    state_input: Agent2State = {
        "cv_text": json.dumps(cv_json, ensure_ascii=False),
        "user_target_text": user_target_text.strip(),
        "resolved_role": "",
        "hyde_document": "",
        "candidate_jobs": [],
        "top_jobs": [],
        "best_score": 0.0,
        "retry_count": 0,
        "feedback": None,
        "source": "",
    }

    agent2_result = agent2_app.invoke(state_input)

    top_jobs = agent2_result.get("top_jobs", [])
    source = agent2_result.get("source", "unknown")
    resolved_role = agent2_result.get("resolved_role", "Unknown")

    print(f"\n✅ Agent 2 completed search:")
    print(f"   • Resolved Target Role: '{resolved_role}'")
    print(f"   • Data Source: {source.upper()}")
    print(f"   • Matched Job Profiles Count: {len(top_jobs)}")

    for idx, job in enumerate(top_jobs, 1):
        score_info = (
            f"(Score: {job.get('rerank_score')})" if "rerank_score" in job else ""
        )
        print(
            f"     [{idx}] {job.get('title')} ({job.get('experience_level')}) {score_info}"
        )
        print(f"         Tech Stack: {', '.join(job.get('primary_tech_stack', []))}")

    return top_jobs


# Allows interactive CLI selection or programmatic auto-selection of the desired job profile from the top candidate matches.
def step_3_select_job(
    top_jobs: List[Dict[str, Any]], auto_select_index: Optional[int] = None
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("🎯 [STEP 3] JOB PROFILE SELECTION FOR CAREER STRATEGY")
    print("=" * 70)

    if not top_jobs:
        raise ValueError("❌ No job profiles available for selection.")

    if auto_select_index is not None:
        idx = max(0, min(auto_select_index, len(top_jobs) - 1))
        chosen_job = top_jobs[idx]
        print(
            f"\n⚡ Auto-selected Job [{idx + 1}]: {chosen_job.get('title')} ({chosen_job.get('experience_level')})"
        )
        return chosen_job

    print("\nSelect the target job for your comprehensive career transition report:")
    for idx, job in enumerate(top_jobs, 1):
        print(f"  [{idx}] {job.get('title')} - Level: {job.get('experience_level')}")
        print(f"      Tech Stack: {', '.join(job.get('primary_tech_stack', []))}")
        responsibilities = job.get("responsibilities", [])
        if responsibilities:
            print(f"      Key Responsibility: {responsibilities[0]}")
        print()

    while True:
        user_choice = input(
            f"Enter valid selection (number between 1 and {len(top_jobs)}): "
        ).strip()

        if not user_choice:
            print(
                f"⚠️ Empty input. Please select a number between 1 and {len(top_jobs)}."
            )
            continue

        try:
            selected_idx = int(user_choice) - 1
            if 0 <= selected_idx < len(top_jobs):
                break
            else:
                print(
                    f"❌ Selection out of range ({user_choice}). Choose strictly between 1 and {len(top_jobs)}."
                )
        except ValueError:
            print(f"❌ Invalid value ('{user_choice}'). Please enter a valid number.")

    chosen_job = top_jobs[selected_idx]
    print(
        f"\n👉 Successfully selected: '{chosen_job.get('title')}' ({chosen_job.get('experience_level')})"
    )

    return chosen_job


# Invokes Agent 3 ReAct loop to generate the full strategic career transition report.
def step_4_generate_career_report(
    cv_json: Dict[str, Any], selected_job: Dict[str, Any]
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("🧠 [STEP 4] GENERATING STRATEGIC CAREER REPORT (AGENT 3 - ReAct)")
    print("=" * 70)

    initial_state: Agent3State = {
        "cv_data": cv_json,
        "selected_job": selected_job,
        "messages": [
            HumanMessage(
                content=(
                    f"Please evaluate the candidate's profile against the selected target job and produce a complete career transition report.\n\n"
                    f"CANDIDATE CV DATA:\n{json.dumps(cv_json, ensure_ascii=False, indent=2)}\n\n"
                    f"TARGET JOB PROFILE:\n{json.dumps(selected_job, ensure_ascii=False, indent=2)}"
                )
            )
        ],
        "loop_step": 0,
        "final_report": None,
    }

    print("\n🔄 Starting autonomous execution for Agent 3...")
    result_state = agent3_graph.invoke(initial_state)

    final_report = result_state.get("final_report")
    if not final_report:
        raise RuntimeError("❌ Agent 3 failed to return a valid final career report.")

    print("\n✅ Strategic career transition report generated successfully!")
    return final_report


# Prints the strategic report to the console and exports it to a structured JSON file.
def display_and_save_report(
    report: Dict[str, Any], output_path: str = "career_report.json"
) -> None:
    print("\n" + "=" * 80)
    print("🏆 FINAL STRATEGIC CAREER TRANSITION REPORT")
    print("=" * 80)

    print(f"\n🎯 TARGET ROLE: {report.get('target_role')}")
    print(f"\n📌 EXECUTIVE SUMMARY:\n{report.get('executive_summary')}")

    gap = report.get("gap_analysis", {})
    print("\n" + "-" * 40)
    print(f"📊 TECHNICAL GAP ANALYSIS (Match: {gap.get('match_percentage')}%)")
    print("-" * 40)
    print(f"• Seniority Fit:       {gap.get('seniority_fit')}")
    print(f"• Matching Skills:     {', '.join(gap.get('matching_skills', []))}")
    print(f"• Missing Skills:      {', '.join(gap.get('missing_skills', []))}")
    print(f"• Transferable Skills: {', '.join(gap.get('transferable_skills', []))}")
    print(f"• Justification:       {gap.get('justification')}")

    roadmap = report.get("learning_roadmap", [])
    print("\n" + "-" * 40)
    print(f"🛣️ PROGRESSIVE LEARNING ROADMAP ({len(roadmap)} Phases)")
    print("-" * 40)
    for phase in roadmap:
        print(
            f"\n[Phase {phase.get('phase_number')}] {phase.get('title')} (~{phase.get('duration_weeks')} weeks)"
        )
        print(f"   • Technologies: {', '.join(phase.get('focus_technologies', []))}")
        print(f"   • Resources:    {', '.join(phase.get('recommended_resources', []))}")
        print(f"   • Milestone:    {phase.get('practical_milestone')}")

    proj = report.get("portfolio_project", {})
    print("\n" + "-" * 40)
    print(f"🏗️ PORTFOLIO PROJECT: {proj.get('project_title')}")
    print("-" * 40)
    print(f"• Business Context:  {proj.get('business_context')}")
    print(f"• Role Alignment:    {proj.get('target_role_alignment')}")
    print(f"• Tech Stack:        {', '.join(proj.get('tech_stack_used', []))}")
    print(f"• Architecture:      {proj.get('architecture_overview')}")
    print("• Key Features:")
    for feat in proj.get("key_features_to_implement", []):
        print(f"   - {feat}")
    print("• Implementation Guide:")
    for step in proj.get("step_by_step_implementation_guide", []):
        print(f"   {step}")
    print(f"• GitHub README Guide:\n  {proj.get('github_readme_highlights')}")

    questions = report.get("interview_preparation", [])
    print("\n" + "-" * 40)
    print(f"🎙️ INTERVIEW PREPARATION ({len(questions)} Questions)")
    print("-" * 40)
    for idx, q in enumerate(questions, 1):
        print(f"\n[{idx}] ({q.get('category')}) {q.get('question')}")
        print("    Ideal Response Key Points:")
        for pt in q.get("ideal_answer_points", []):
            print(f"     - {pt}")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Full report saved to: '{output_path}'")
    except Exception as e:
        print(f"\n⚠️ Could not save report to file: {e}")


# Executes the complete end-to-end multi-agent career transition pipeline.
def run_career_pipeline(
    cv_input: str,
    user_target_text: str = "",
    is_pdf: bool = True,
    auto_select_job_index: Optional[int] = None,
) -> Dict[str, Any]:
    print("\n" + "#" * 80)
    print("🚀 STARTING MULTI-AGENT CAREER ADVISOR PIPELINE")
    print("#" * 80)

    cv_json = step_1_extract_cv(cv_input, is_pdf=is_pdf)
    top_jobs = step_2_search_jobs(cv_json, user_target_text=user_target_text)
    selected_job = step_3_select_job(top_jobs, auto_select_index=auto_select_job_index)
    final_report = step_4_generate_career_report(cv_json, selected_job)
    display_and_save_report(final_report)

    return final_report


if __name__ == "__main__":
    sample_raw_cv = """john doe pine street austin tx 78701 555 8923000 johndoedevexamplecom professional summary passionate software engineer around 2 years experience
     full stack web development backend engineering restful apis relational databases containerization microservices solid understanding object oriented programming
      software design patterns automated testing clean code practices proficent building scalable web applications python fastapi django modern frontend interfaces
       react typescript experienced working agile scrum teams managing database schemas performance tuning git version control technical skills programming languages
        python javascript typescript sql html5 css3 backend frameworks fastapi django flask nodejs express frontend react redux tailwindcss databases postgresql mysql
         sqlite redis orm tools sqlalchemy alembic containerization tools docker docker compose cicd github actions testing pytest unittest postman version control
          git github operating systems ubuntu linux macos windows tools visual studio code dbeaver insomnia work history software developer 062022 current techflow 
          solutions austin tx designed developed high performance asynchronous rest apis python fastapi postgresql reducing average api latency 35 percent implemented 
          user authentication authorization jwt tokens oauth2 managed database migrations schema optimizations using sqlalchemy alembic integrated redis caching layer 
          frequently accessed endpoints improving read throughput containerized backend microservices frontend client docker docker compose local development automated
           testing linting workflows configuring github actions cicd pipelines built interactive frontend dashboards user management modules using react typescript
            tailwindcss collaborated cross functional team daily standups sprint planning code reviews junior software developer intern 012022 052022 cloudbyte labs
             austin tx assisted senior engineers building crud endpoints flask sqlite wrote comprehensive unit integration tests using pytest achieving 85 percent test
              coverage fixed bugs legacy backend modules maintained technical documentation education bachelor science computer science 2018 2022 university texas 
              austin tx"""

    user_preference = "I want to transition into Cloud Engineering and DevOps, focusing on AWS and Kubernetes."

    run_career_pipeline(
        cv_input=sample_raw_cv,
        user_target_text=user_preference,
        is_pdf=False,
        auto_select_job_index=None,
    )
