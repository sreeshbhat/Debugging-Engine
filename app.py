import pandas as pd
import streamlit as st

from services.llm_service import LLMService
from services.mongo_service import SubmissionRepository
from services.scoring_service import evaluate_student_prompt, fallback_score_payload
from utils.helpers import (
    CHALLENGES_PATH,
    RESULTS_PATH,
    badge_from_score,
    deterministic_provider_bucket,
    filter_student_submissions,
    format_attempt_time,
    get_secret,
    grade_from_score,
    leaderboard_dataframe,
    load_challenges,
    now_iso,
    submissions_to_csv_bytes,
)


st.set_page_config(page_title="BugFix Prompt Arena", page_icon="🛠️", layout="wide")


def init_session_state() -> None:
    st.session_state.setdefault("started", False)
    st.session_state.setdefault("selected_challenge_id", None)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("teacher_unlocked", False)
    st.session_state.setdefault("student_name", "")
    st.session_state.setdefault("student_roll", "")
    st.session_state.setdefault("prompt_input", "")
    st.session_state.setdefault("show_improved_prompt", False)
    st.session_state.setdefault("reveal_actual_fix", False)
    st.session_state.setdefault("gemini_api_key_input", "")
    st.session_state.setdefault("groq_api_key_input", "")
    st.session_state.setdefault("openai_api_key_input", "")


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .arena-badge {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.9rem;
            background: linear-gradient(135deg, #102542, #f87060);
            color: white;
            margin-bottom: 0.5rem;
        }
        .arena-grade {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.9rem;
            background: #f4f1de;
            color: #202c39;
            border: 1px solid #d9c6a5;
            margin-bottom: 0.5rem;
            margin-right: 0.5rem;
        }
        .arena-card {
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1rem;
            background: #fffdf8;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_repository() -> SubmissionRepository:
    return SubmissionRepository(get_secret("MONGODB_URI"), RESULTS_PATH)


def unlock_teacher_mode() -> bool:
    expected_password = get_secret("TEACHER_PASSWORD", "admin123")
    entered_password = st.sidebar.text_input("Teacher password", type="password")

    if entered_password and entered_password == expected_password:
        st.session_state.teacher_unlocked = True
    elif entered_password:
        st.session_state.teacher_unlocked = False
        st.sidebar.error("Incorrect teacher password.")

    return st.session_state.teacher_unlocked


def get_api_key_overrides() -> dict[str, str]:
    overrides = {}
    if st.session_state.gemini_api_key_input.strip():
        overrides["Gemini"] = st.session_state.gemini_api_key_input.strip()
    if st.session_state.groq_api_key_input.strip():
        overrides["Groq"] = st.session_state.groq_api_key_input.strip()
    if st.session_state.openai_api_key_input.strip():
        overrides["OpenAI"] = st.session_state.openai_api_key_input.strip()
    return overrides


def render_sidebar(
    challenges: list[dict],
    repository: SubmissionRepository,
) -> tuple[str, bool, dict[str, str]]:
    st.sidebar.title("Arena Settings")

    with st.sidebar.expander("Session API Keys", expanded=False):
        st.caption("These override server keys only for your current session.")
        st.text_input("Gemini API key", type="password", key="gemini_api_key_input")
        st.text_input("Groq API key", type="password", key="groq_api_key_input")
        st.text_input("OpenAI API key", type="password", key="openai_api_key_input")

    api_key_overrides = get_api_key_overrides()
    provider_mode = st.sidebar.selectbox(
        "Scoring provider mode",
        ["Auto Balanced", "Gemini", "Groq", "OpenAI"],
    )
    provider_details = LLMService.get_provider_details(provider_mode, api_key_overrides)
    st.sidebar.caption(
        f"Route: `{provider_details['provider']}` | Backing model(s): `{provider_details['model']}`"
    )
    if provider_mode == "Auto Balanced":
        configured = LLMService.configured_providers(api_key_overrides)
        if configured:
            st.sidebar.caption("Configured providers: " + ", ".join(configured))
        else:
            st.sidebar.warning("No LLM API keys are configured yet.")
    else:
        st.sidebar.caption(
            "API key status: " + ("configured" if provider_details["configured"] else "missing")
        )

    teacher_enabled = st.sidebar.toggle("Teacher mode", value=False)
    is_teacher = unlock_teacher_mode() if teacher_enabled else False

    if teacher_enabled and is_teacher:
        st.sidebar.success("Teacher mode unlocked.")

    st.sidebar.divider()
    st.sidebar.caption(f"Storage: {repository.status_message()}")

    if st.session_state.student_name and st.session_state.student_roll:
        st.sidebar.info(
            f"Student: {st.session_state.student_name}\n\nRoll: {st.session_state.student_roll}"
        )

    if challenges:
        selected_title = st.sidebar.selectbox(
            "Choose challenge",
            options=[challenge["title"] for challenge in challenges],
            index=_selected_challenge_index(challenges),
        )
        for challenge in challenges:
            if challenge["title"] == selected_title:
                st.session_state.selected_challenge_id = challenge["id"]
                break

    if st.sidebar.button("Try Again"):
        reset_attempt_state()
        st.rerun()

    return provider_mode, is_teacher, api_key_overrides


def _selected_challenge_index(challenges: list[dict]) -> int:
    selected_id = st.session_state.get("selected_challenge_id")
    for index, challenge in enumerate(challenges):
        if challenge["id"] == selected_id:
            return index
    if challenges:
        st.session_state.selected_challenge_id = challenges[0]["id"]
    return 0


def reset_attempt_state() -> None:
    st.session_state.last_result = None
    st.session_state.prompt_input = ""
    st.session_state.show_improved_prompt = False
    st.session_state.reveal_actual_fix = False


def render_home() -> None:
    st.title("BugFix Prompt Arena")
    st.subheader("Practice writing better AI debugging prompts")
    st.write(
        "Students do not fix the bug directly here. They write the prompt they would send to an AI coding assistant, "
        "and the arena scores how effective that debugging prompt is."
    )
    st.write(
        "Before entering the challenge arena, each student must register with a name and roll number. "
        "That identity is used for the leaderboard, attempt history, and provider load balancing."
    )

    st.markdown(
        """
        **How the arena works**
        1. Enter your name and roll number.
        2. Read one buggy scenario at a time.
        3. Write a strong debugging prompt with context, the exact error, constraints, and expected output.
        4. Review the score, badge, weaknesses, and improved prompt.
        """
    )

    with st.form("student_entry_form"):
        student_name = st.text_input("Student name", value=st.session_state.student_name)
        student_roll = st.text_input("Roll number", value=st.session_state.student_roll)
        start_clicked = st.form_submit_button("Enter Arena", type="primary", use_container_width=True)

    if start_clicked:
        if not student_name.strip():
            st.error("Student name is required.")
            return
        if not student_roll.strip():
            st.error("Roll number is required.")
            return

        st.session_state.student_name = student_name.strip()
        st.session_state.student_roll = student_roll.strip()
        st.session_state.started = True
        reset_attempt_state()
        st.rerun()


def resolve_provider(
    provider_mode: str,
    challenge_id: str,
    api_key_overrides: dict[str, str],
) -> str:
    if provider_mode != "Auto Balanced":
        if not LLMService.provider_has_key(provider_mode, api_key_overrides):
            raise ValueError(f"{provider_mode} is selected but its API key is missing.")
        return provider_mode

    configured = LLMService.configured_providers(api_key_overrides)
    return deterministic_provider_bucket(st.session_state.student_roll, challenge_id, configured)


def render_challenge(
    challenge: dict,
    provider_mode: str,
    is_teacher: bool,
    repository: SubmissionRepository,
    api_key_overrides: dict[str, str],
) -> None:
    st.title("BugFix Prompt Arena")
    st.caption(f"{challenge['difficulty']} | {challenge['category']}")
    st.header(challenge["title"])

    assigned_provider = "Unavailable"
    try:
        assigned_provider = resolve_provider(provider_mode, challenge["id"], api_key_overrides)
    except Exception as exc:
        st.warning(str(exc))

    top_col1, top_col2, top_col3 = st.columns(3)
    top_col1.markdown(
        f"<div class='arena-card'><strong>Student</strong><br>{st.session_state.student_name}</div>",
        unsafe_allow_html=True,
    )
    top_col2.markdown(
        f"<div class='arena-card'><strong>Roll Number</strong><br>{st.session_state.student_roll}</div>",
        unsafe_allow_html=True,
    )
    top_col3.markdown(
        f"<div class='arena-card'><strong>Assigned Provider</strong><br>{assigned_provider}</div>",
        unsafe_allow_html=True,
    )

    details_col, form_col = st.columns([1.2, 1], gap="large")

    with details_col:
        st.write("**Error message**")
        st.code(challenge["error_message"], language="text")

        st.write("**Buggy code snippet**")
        st.code(challenge["buggy_code"], language="python")

        st.write("**Expected behavior**")
        st.info(challenge["expected_behavior"])

        st.write("**Current broken behavior**")
        st.warning(challenge["broken_behavior"])

        if is_teacher:
            if st.button("Reveal Actual Fix", key="reveal_actual_fix_button"):
                st.session_state.reveal_actual_fix = not st.session_state.reveal_actual_fix
            if st.session_state.reveal_actual_fix:
                st.write("**Root cause**")
                st.write(challenge["root_cause"])
                st.write("**Correct fix**")
                st.write(challenge["correct_fix"])
                st.write("**Teacher notes**")
                st.write(challenge["teacher_notes"])

    with form_col:
        st.write("**Student Submission**")
        with st.form("prompt_submission_form"):
            prompt_text = st.text_area(
                "Write the prompt you will give to an AI coding assistant",
                key="prompt_input",
                height=260,
                placeholder=(
                    "I am debugging a Python Streamlit app. The exact error is ... Please inspect the root cause "
                    "step by step, keep the fix minimal, preserve existing behavior, and explain how to verify it."
                ),
            )
            submitted = st.form_submit_button("Submit for rating", type="primary", use_container_width=True)

        if submitted:
            result = handle_submission(
                challenge=challenge,
                prompt_text=prompt_text,
                provider_mode=provider_mode,
                is_teacher=is_teacher,
                repository=repository,
                api_key_overrides=api_key_overrides,
            )
            st.session_state.last_result = result

        action_col1, action_col2 = st.columns(2)
        if action_col1.button("Show Improved Prompt", use_container_width=True):
            st.session_state.show_improved_prompt = not st.session_state.show_improved_prompt
        if action_col2.button("Try Again", use_container_width=True):
            reset_attempt_state()
            st.rerun()

    if st.session_state.last_result:
        render_result(st.session_state.last_result)

    render_attempt_history(repository.fetch_submissions())


def handle_submission(
    challenge: dict,
    prompt_text: str,
    provider_mode: str,
    is_teacher: bool,
    repository: SubmissionRepository,
    api_key_overrides: dict[str, str],
) -> dict:
    if not st.session_state.student_name.strip():
        st.error("Please enter the arena using your name before submitting.")
        return {}

    if not st.session_state.student_roll.strip():
        st.error("Please enter the arena using your roll number before submitting.")
        return {}

    if not prompt_text.strip():
        st.error("Please write a debugging prompt before submitting.")
        return {}

    try:
        provider = resolve_provider(provider_mode, challenge["id"], api_key_overrides)
    except Exception as exc:
        st.error(str(exc))
        return {}

    with st.spinner(f"Evaluating prompt quality with {provider}..."):
        try:
            result = evaluate_student_prompt(
                challenge=challenge,
                student_prompt=prompt_text,
                provider=provider,
                teacher_mode=is_teacher,
                api_key_overrides=api_key_overrides,
            )
        except Exception as exc:
            st.warning(
                "Automatic scoring was unavailable. A fallback response was generated instead."
            )
            result = fallback_score_payload(str(exc))

    score = result.get("total_score", 0)
    result["grade"] = grade_from_score(score)
    result["badge"] = badge_from_score(score)
    result["selected_provider"] = provider

    submission = {
        "student_name": st.session_state.student_name.strip(),
        "student_roll": st.session_state.student_roll.strip(),
        "student_email": "",
        "challenge_id": challenge["id"],
        "challenge_title": challenge["title"],
        "prompt_text": prompt_text.strip(),
        "selected_provider": provider,
        "score": score,
        "grade": result["grade"],
        "badge": result["badge"],
        "ai_feedback_json": result,
        "created_at": now_iso(),
    }

    backend_used = repository.save_submission(submission)
    st.success(f"Prompt rated successfully. Saved using {backend_used}.")
    return result


def render_result(result: dict) -> None:
    if not result:
        return

    st.divider()
    st.subheader("Score Card")

    score = result.get("total_score", 0)
    grade = result.get("grade", "Poor")
    badge = result.get("badge", badge_from_score(score))
    likely = result.get("would_this_likely_solve_the_bug", "Partially")

    st.markdown(f"<span class='arena-grade'>{grade}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='arena-badge'>{badge}</span>", unsafe_allow_html=True)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Total Score", f"{score}/100")
    metric_col2.metric("Grade", grade)
    metric_col3.metric("Likely to solve?", likely)
    metric_col4.metric("Scored By", result.get("selected_provider", "Unknown"))

    st.progress(score / 100)

    breakdown_rows = []
    for key, section in result.get("breakdown", {}).items():
        label = key.replace("_", " ").title()
        breakdown_rows.append(
            {
                "Category": label,
                "Score": section.get("score", 0),
                "Feedback": section.get("feedback", ""),
            }
        )

    st.write("**Breakdown**")
    st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

    st.write("**Strengths**")
    strengths = result.get("strengths") or ["No major strengths were identified."]
    for item in strengths:
        st.write(f"- {item}")

    st.write("**Weaknesses**")
    weaknesses = result.get("weaknesses") or ["No major weaknesses were identified."]
    for item in weaknesses:
        st.write(f"- {item}")

    if st.session_state.show_improved_prompt:
        st.write("**Improved Prompt**")
        st.code(result.get("improved_prompt", ""), language="text")

    st.write("**Evaluator Reason**")
    st.info(result.get("reason", ""))


def render_leaderboard(submissions: list[dict]) -> None:
    st.divider()
    st.subheader("Leaderboard")

    leaderboard = leaderboard_dataframe(submissions)
    if leaderboard.empty:
        st.info("No submissions yet.")
        return

    st.dataframe(leaderboard, use_container_width=True, hide_index=True)


def render_attempt_history(submissions: list[dict]) -> None:
    st.divider()
    st.subheader("Attempt History")

    student_attempts = filter_student_submissions(submissions, st.session_state.student_roll)
    if not student_attempts:
        st.info("No attempts recorded for this student yet.")
        return

    rows = []
    for item in sorted(student_attempts, key=lambda record: record.get("created_at", ""), reverse=True):
        rows.append(
            {
                "Challenge": item.get("challenge_title", ""),
                "Score": item.get("score", 0),
                "Grade": item.get("grade", ""),
                "Badge": item.get("badge", badge_from_score(item.get("score", 0))),
                "Provider": item.get("selected_provider", ""),
                "Attempt Time": format_attempt_time(item.get("created_at", "")),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_teacher_dashboard(submissions: list[dict]) -> None:
    st.divider()
    st.subheader("Teacher Dashboard")

    if not submissions:
        st.info("No submissions available yet.")
        return

    average_score = round(sum(item.get("score", 0) for item in submissions) / len(submissions), 2)
    unique_students = len({item.get("student_roll", "") for item in submissions if item.get("student_roll")})
    stat_col1, stat_col2 = st.columns(2)
    stat_col1.metric("Average Score", average_score)
    stat_col2.metric("Unique Students", unique_students)

    st.write("**All Student Submissions**")
    submissions_frame = pd.DataFrame(submissions)
    st.dataframe(submissions_frame, use_container_width=True, hide_index=True)

    st.download_button(
        "Download results as CSV",
        data=submissions_to_csv_bytes(submissions),
        file_name="bugfix_prompt_arena_results.csv",
        mime="text/csv",
    )


def main() -> None:
    init_session_state()
    apply_theme()

    challenges = load_challenges(CHALLENGES_PATH)
    repository = get_repository()
    provider_mode, is_teacher, api_key_overrides = render_sidebar(challenges, repository)
    if not challenges:
        st.title("BugFix Prompt Arena")
        st.error(
            "Challenge data could not be loaded. Please check whether data/challenges.json exists and contains valid JSON."
        )
        return

    if not st.session_state.started:
        render_home()
    else:
        selected_id = st.session_state.get("selected_challenge_id") or challenges[0]["id"]
        current_challenge = next(
            (challenge for challenge in challenges if challenge["id"] == selected_id),
            challenges[0],
        )
        render_challenge(
            current_challenge,
            provider_mode,
            is_teacher,
            repository,
            api_key_overrides,
        )

    submissions = repository.fetch_submissions()
    render_leaderboard(submissions)

    if is_teacher:
        render_teacher_dashboard(submissions)


if __name__ == "__main__":
    main()
