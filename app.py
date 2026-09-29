import streamlit as st
import tempfile
import os
import json
from pathlib import Path
from langchain_core.messages import HumanMessage

from agent_1 import Agent1CVExtractor
from agent_2_pro import agent2_app, Agent2State
from agent3 import agent3_graph, Agent3State

st.set_page_config(
    page_title="AI Career Advisor Multi-Agent",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initializes Agent 1 once and caches the loaded fine-tuned model in GPU memory.
@st.cache_resource(show_spinner=False)
def load_agent1():
    return Agent1CVExtractor()


if "stage" not in st.session_state:
    st.session_state.stage = 1
if "cv_data" not in st.session_state:
    st.session_state.cv_data = None
if "job_profiles" not in st.session_state:
    st.session_state.job_profiles = None
if "resolved_role" not in st.session_state:
    st.session_state.resolved_role = None
if "data_source" not in st.session_state:
    st.session_state.data_source = None
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "career_report" not in st.session_state:
    st.session_state.career_report = None


# Resets the entire application state and restarts the workflow.
def reset_pipeline():
    st.session_state.stage = 1
    st.session_state.cv_data = None
    st.session_state.job_profiles = None
    st.session_state.resolved_role = None
    st.session_state.data_source = None
    st.session_state.selected_job = None
    st.session_state.career_report = None
    st.rerun()


with st.sidebar:
    st.header("⚙️ Configuration & Input")

    uploaded_file = st.file_uploader(
        "Upload CV (PDF format)",
        type=["pdf"],
        help="Upload the PDF file to be processed by Agent 1.",
    )

    cv_raw_text = st.text_area(
        "Or paste raw CV text:",
        height=150,
        placeholder="Paste CV text here if you do not have a PDF...",
    )

    user_target = st.text_input(
        "Career Preference / Target Role (Optional):",
        placeholder="e.g. I want to transition into Cloud Engineering and DevOps, focusing on AWS and Kubernetes.",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_analysis = st.button(
            "🚀 Analyze CV", type="primary", use_container_width=True
        )
    with col_btn2:
        if st.button("🔄 Reset", use_container_width=True):
            reset_pipeline()

st.title("💼 AI Career Transition Advisor")
st.markdown(
    "**Autonomous Multi-Agent System** for technical profile restructuring, "
    "target job matching, and full career transition strategy compilation."
)
st.divider()

if start_analysis:
    if not uploaded_file and not cv_raw_text.strip():
        st.warning("⚠️ Please upload a PDF file or provide the raw CV text.")
    else:
        st.session_state.stage = 2
        st.session_state.uploaded_file = uploaded_file
        st.session_state.cv_raw_text = cv_raw_text
        st.session_state.user_target = user_target
        st.session_state.cv_data = None
        st.session_state.job_profiles = None
        st.session_state.selected_job = None
        st.session_state.career_report = None
        st.rerun()

if st.session_state.stage >= 2:
    st.subheader("📋 Step 1: Extracted Technical Profile (Agent 1)")

    if st.session_state.cv_data is None:
        with st.status(
            "🧠 Loading model and extracting profile on GPU (RTX 4050)...",
            expanded=True,
        ) as status:
            agent1 = load_agent1()

            if st.session_state.uploaded_file is not None:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(st.session_state.uploaded_file.read())
                    temp_pdf_path = tmp_file.name
                st.write("📄 Extracting text from PDF...")
                extracted_data = agent1.run(temp_pdf_path, is_pdf=True)
                os.remove(temp_pdf_path)
            else:
                st.write("📝 Processing raw CV text...")
                extracted_data = agent1.run(st.session_state.cv_raw_text, is_pdf=False)

            st.session_state.cv_data = extracted_data
            status.update(
                label="✅ CV successfully parsed and structured!",
                state="complete",
                expanded=False,
            )

    cv = st.session_state.cv_data
    if cv:
        role = cv.get("Candidate_Role") or cv.get("candidate_role", "N/A")
        level = cv.get("Experience_Level") or cv.get("experience_level", "N/A")
        years = cv.get("Years_Of_Experience") or cv.get("years_of_experience", 0)
        summary = cv.get("Clean_Summary") or cv.get(
            "clean_summary", "No summary available."
        )
        skills = cv.get("Primary_Skills") or cv.get("primary_skills", [])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Identified Role", role)
        with col2:
            st.metric("Seniority Level", level)
        with col3:
            st.metric("Years of Experience", f"{years} yrs")

        st.markdown("**Executive Profile Summary:**")
        st.info(summary)

        st.markdown("**Extracted Technical Skills:**")
        if skills:
            skills_html = " ".join(
                [
                    f'<span style="background-color: #1E293B; color: #38BDF8; padding: 4px 10px; margin: 3px; border-radius: 12px; font-weight: 500; display: inline-block;">{s}</span>'
                    for s in skills
                ]
            )
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.write("No specific technical skills identified.")

    st.divider()

if st.session_state.stage >= 2 and st.session_state.cv_data:
    st.subheader("🎯 Step 2: Target Job Selection for Transition (Agent 2)")

    if st.session_state.job_profiles is None:
        with st.status(
            "🔍 Running Agent 2: Hybrid Search (Qdrant), HyDE & Reranking...",
            expanded=True,
        ) as status:
            state_input: Agent2State = {
                "cv_text": json.dumps(st.session_state.cv_data, ensure_ascii=False),
                "user_target_text": (st.session_state.user_target or "").strip(),
                "resolved_role": "",
                "hyde_document": "",
                "candidate_jobs": [],
                "top_jobs": [],
                "best_score": 0.0,
                "retry_count": 0,
                "feedback": None,
                "source": "",
            }

            st.write("⚙️ Executing Agent 2 decision graph...")
            agent2_result = agent2_app.invoke(state_input)

            jobs_found = (
                agent2_result.get("top_jobs")
                or agent2_result.get("reranked_job_profiles")
                or agent2_result.get("final_job_profiles")
                or []
            )

            resolved_r = (
                agent2_result.get("resolved_role")
                or agent2_result.get("resolved_target_role")
                or "Target Role"
            )

            src = (
                agent2_result.get("source")
                or agent2_result.get("data_source")
                or "LOCAL"
            )

            st.session_state.job_profiles = jobs_found
            st.session_state.resolved_role = resolved_r
            st.session_state.data_source = str(src).upper()

            status.update(
                label=f"✅ Retrieved {len(st.session_state.job_profiles)} job profiles ({st.session_state.data_source})!",
                state="complete",
                expanded=False,
            )

    jobs = st.session_state.job_profiles
    if jobs:
        st.write(
            f"Resolved Target Role: **{st.session_state.resolved_role}** (Data Source: `{st.session_state.data_source}`)"
        )

        cols = st.columns(len(jobs))
        for idx, job in enumerate(jobs):
            job_dict = job if isinstance(job, dict) else job.model_dump()

            title = job_dict.get("title", "Job Title")
            level = job_dict.get("experience_level", "N/A")
            stack = job_dict.get("primary_tech_stack", [])
            resps = job_dict.get("responsibilities", [])

            with cols[idx]:
                st.markdown(f"### {title}")
                st.caption(f"Level: **{level}**")

                st.markdown("**Tech Stack:**")
                stack_badges = " ".join(
                    [
                        f'<span style="background-color: #334155; color: #F8FAFC; padding: 2px 6px; margin: 2px; border-radius: 6px; font-size: 12px; display: inline-block;">{t}</span>'
                        for t in stack
                    ]
                )
                st.markdown(stack_badges, unsafe_allow_html=True)

                st.markdown("**Responsibilities:**")
                for r in resps[:3]:
                    st.markdown(f"- <small>{r}</small>", unsafe_allow_html=True)

                st.write("")
                is_selected = st.session_state.selected_job == job_dict
                btn_label = "✅ Selected" if is_selected else "🎯 Choose this Role"
                btn_type = "primary" if is_selected else "secondary"

                if st.button(
                    btn_label,
                    key=f"job_btn_{idx}",
                    type=btn_type,
                    use_container_width=True,
                ):
                    st.session_state.selected_job = job_dict
                    st.session_state.stage = 3
                    st.session_state.career_report = None
                    st.rerun()

    if st.session_state.selected_job:
        st.success(
            f"👉 Selected Job Profile: **{st.session_state.selected_job.get('title')} ({st.session_state.selected_job.get('experience_level')})**"
        )

    st.divider()

if st.session_state.stage >= 3 and st.session_state.selected_job:
    st.subheader("🏆 Step 3: Strategic Career Transition Report (Agent 3 - ReAct)")

    if st.session_state.career_report is None:
        with st.status(
            "🧠 Agent 3 running autonomous ReAct loop (gap analysis + web research)...",
            expanded=True,
        ) as status:
            initial_react_state: Agent3State = {
                "cv_data": st.session_state.cv_data,
                "selected_job": st.session_state.selected_job,
                "messages": [
                    HumanMessage(
                        content=(
                            f"Please evaluate the candidate's profile against the selected target job and produce a complete career transition report.\n\n"
                            f"CANDIDATE CV DATA:\n{json.dumps(st.session_state.cv_data, ensure_ascii=False, indent=2)}\n\n"
                            f"TARGET JOB PROFILE:\n{json.dumps(st.session_state.selected_job, ensure_ascii=False, indent=2)}"
                        )
                    )
                ],
                "loop_step": 0,
                "final_report": None,
            }

            st.write("🔄 Invoking autonomous tools and synthesizing report...")
            react_output = agent3_graph.invoke(initial_react_state)

            st.session_state.career_report = react_output.get("final_report")
            status.update(
                label="🎉 Strategic Career Report compiled successfully!",
                state="complete",
                expanded=False,
            )

    report = st.session_state.career_report
    if report:
        st.markdown("### 📌 Executive Summary")
        st.info(report.get("executive_summary", ""))

        st.markdown("### 📊 Technical Gap Analysis")
        gap = report.get("gap_analysis", {})
        match_score = gap.get("match_percentage", 0)

        col_m1, col_m2 = st.columns([1, 3])
        with col_m1:
            st.metric("Technical Match", f"{match_score}%")
            st.caption(f"Seniority Fit: **{gap.get('seniority_fit', 'N/A')}**")
        with col_m2:
            st.progress(match_score / 100.0)
            st.markdown(f"**Justification:** {gap.get('justification', '')}")

        col_sk1, col_sk2 = st.columns(2)
        with col_sk1:
            st.markdown("**✅ Matching Skills:**")
            matching = gap.get("matching_skills", [])
            m_html = (
                " ".join(
                    [
                        f'<span style="background-color: #064E3B; color: #6EE7B7; padding: 3px 8px; margin: 2px; border-radius: 8px; font-size: 13px; display: inline-block;">{s}</span>'
                        for s in matching
                    ]
                )
                if matching
                else "None"
            )
            st.markdown(m_html, unsafe_allow_html=True)

        with col_sk2:
            st.markdown("**⚠️ Missing Skills (To Acquire):**")
            missing = gap.get("missing_skills", [])
            mis_html = (
                " ".join(
                    [
                        f'<span style="background-color: #7F1D1D; color: #FCA5A5; padding: 3px 8px; margin: 2px; border-radius: 8px; font-size: 13px; display: inline-block;">{s}</span>'
                        for s in missing
                    ]
                )
                if missing
                else "None"
            )
            st.markdown(mis_html, unsafe_allow_html=True)

        st.divider()

        st.markdown("### 🛣️ Progressive Learning Roadmap")
        roadmap = report.get("learning_roadmap", [])
        for phase in roadmap:
            p_num = phase.get("phase_number", "")
            p_title = phase.get("title", f"Phase {p_num}")
            p_weeks = phase.get("duration_weeks", 0)

            with st.expander(f"**{p_title}** (~{p_weeks} weeks)", expanded=True):
                st.markdown(
                    f"**Focus Technologies:** `{', '.join(phase.get('focus_technologies', []))}`"
                )
                st.markdown(
                    f"**Practical Milestone:** {phase.get('practical_milestone', '')}"
                )
                st.markdown("**Recommended Resources:**")
                for res in phase.get("recommended_resources", []):
                    st.markdown(f"- {res}")

        st.divider()

        st.markdown("### 🏗️ Recommended Portfolio Project")
        proj = report.get("portfolio_project", {})
        with st.container(border=True):
            st.subheader(f"💡 {proj.get('project_title', 'Portfolio Project')}")
            st.markdown(f"**Business Context:** {proj.get('business_context', '')}")
            st.markdown(
                f"**Target Tech Stack:** `{', '.join(proj.get('tech_stack_used', []))}`"
            )
            st.markdown(f"**Architecture:** {proj.get('architecture_overview', '')}")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("**Key Features:**")
                for feat in proj.get("key_features_to_implement", []):
                    st.markdown(f"- {feat}")
            with col_p2:
                st.markdown("**Implementation Guide:**")
                for s_idx, step in enumerate(
                    proj.get("step_by_step_implementation_guide", []), 1
                ):
                    st.markdown(f"**{s_idx}.** {step}")

            st.info(
                f"**GitHub README Highlights:** {proj.get('github_readme_highlights', '')}"
            )

        st.divider()

        st.markdown("### 🎙️ Technical Interview Preparation & Scenarios")
        interviews = report.get("interview_preparation", [])
        for i_idx, item in enumerate(interviews, 1):
            cat = item.get("category", "General")
            q = item.get("question", "")
            with st.expander(f"**Q{i_idx}: [{cat}]** {q}"):
                st.markdown("**Ideal Answer Key Points:**")
                for pt in item.get("ideal_answer_points", []):
                    st.markdown(f"- {pt}")

        st.divider()

        st.download_button(
            label="💾 Download Complete Strategic Report (JSON)",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="career_transition_report.json",
            mime="application/json",
            type="primary",
        )
