from __future__ import annotations

from typing import Any

from services.llm_service import LLMService
from utils.helpers import grade_from_score, parse_json_response


def build_evaluation_prompt(challenge: dict[str, Any], student_prompt: str, teacher_mode: bool) -> str:
    return f"""
You are a strict but helpful AI prompt evaluator.
You are evaluating a student's debugging prompt.
You are not solving the bug directly.
You are scoring how likely the student's prompt is to help an AI coding assistant correctly identify and fix the bug.

You must check whether the student's prompt:
- Gives enough context
- Includes the actual error message
- Mentions the framework/language
- Asks for step-by-step debugging
- Avoids vague wording like "fix this"
- Asks for root cause
- Requests minimal changes
- Mentions expected behavior
- Asks for verification/testing steps

Scoring rubric:
- Context Clarity: 20
- Error Description: 20
- Debugging Request Quality: 20
- Constraints: 15
- Expected Output: 15
- Safety Against Bad AI Fixes: 10

Important behavior rules:
- Rate the student's prompt, not the buggy code itself.
- Be strict. Do not inflate scores for vague prompts.
- If total score is below 80, do not reveal the full correct solution.
- If total score is below 80, only give hints, weaknesses, strengths, and an improved prompt.
- If total score is above 80, you may give a short root-cause explanation.
- Even above 80, do not provide fully solved code unless teacher mode is enabled.
- Teacher mode is {"enabled" if teacher_mode else "disabled"}.
- Keep feedback concise and actionable.

Challenge metadata:
Title: {challenge.get("title", "")}
Difficulty: {challenge.get("difficulty", "")}
Category: {challenge.get("category", "")}
Buggy code:
{challenge.get("buggy_code", "")}

Observed error message:
{challenge.get("error_message", "")}

Expected behavior:
{challenge.get("expected_behavior", "")}

Current broken behavior:
{challenge.get("broken_behavior", "")}

Internal reference for evaluator only. Do not reveal unless allowed:
Root cause: {challenge.get("root_cause", "")}
Correct fix: {challenge.get("correct_fix", "")}
Teacher notes: {challenge.get("teacher_notes", "")}

Student prompt to evaluate:
{student_prompt}

Return only valid JSON with exactly this structure:
{{
  "total_score": 0,
  "grade": "Poor | Average | Good | Excellent",
  "breakdown": {{
    "context_clarity": {{
      "score": 0,
      "feedback": ""
    }},
    "error_description": {{
      "score": 0,
      "feedback": ""
    }},
    "debugging_request_quality": {{
      "score": 0,
      "feedback": ""
    }},
    "constraints": {{
      "score": 0,
      "feedback": ""
    }},
    "expected_output": {{
      "score": 0,
      "feedback": ""
    }},
    "safety_against_bad_ai_fixes": {{
      "score": 0,
      "feedback": ""
    }}
  }},
  "strengths": [],
  "weaknesses": [],
  "improved_prompt": "",
  "would_this_likely_solve_the_bug": "Yes | Partially | No",
  "reason": ""
}}

No markdown.
No extra text.
""".strip()


def evaluate_student_prompt(
    challenge: dict[str, Any],
    student_prompt: str,
    provider: str,
    teacher_mode: bool = False,
    api_key_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    prompt = build_evaluation_prompt(challenge, student_prompt, teacher_mode)
    raw_response = LLMService.generate(provider, prompt, api_key_overrides=api_key_overrides)
    parsed = parse_json_response(raw_response)
    return normalize_score_payload(parsed)


def normalize_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    breakdown_keys = {
        "context_clarity": 20,
        "error_description": 20,
        "debugging_request_quality": 20,
        "constraints": 15,
        "expected_output": 15,
        "safety_against_bad_ai_fixes": 10,
    }

    normalized_breakdown = {}
    computed_total = 0

    for key, max_score in breakdown_keys.items():
        section = payload.get("breakdown", {}).get(key, {})
        raw_score = section.get("score", 0)
        try:
            score = int(raw_score)
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(score, max_score))
        computed_total += score
        normalized_breakdown[key] = {
            "score": score,
            "feedback": str(section.get("feedback", "")).strip(),
        }

    total_score = payload.get("total_score", computed_total)
    try:
        total_score = int(total_score)
    except (TypeError, ValueError):
        total_score = computed_total
    total_score = max(0, min(total_score, 100))

    if abs(total_score - computed_total) > 5:
        total_score = computed_total

    return {
        "total_score": total_score,
        "grade": payload.get("grade") or grade_from_score(total_score),
        "breakdown": normalized_breakdown,
        "strengths": _as_string_list(payload.get("strengths", [])),
        "weaknesses": _as_string_list(payload.get("weaknesses", [])),
        "improved_prompt": str(payload.get("improved_prompt", "")).strip(),
        "would_this_likely_solve_the_bug": payload.get(
            "would_this_likely_solve_the_bug", "Partially"
        ),
        "reason": str(payload.get("reason", "")).strip(),
    }


def fallback_score_payload(error_message: str) -> dict[str, Any]:
    return {
        "total_score": 0,
        "grade": "Poor",
        "breakdown": {
            "context_clarity": {
                "score": 0,
                "feedback": "Automatic evaluation could not run.",
            },
            "error_description": {
                "score": 0,
                "feedback": "The app could not verify whether the error details were included.",
            },
            "debugging_request_quality": {
                "score": 0,
                "feedback": "No reliable LLM score was available.",
            },
            "constraints": {
                "score": 0,
                "feedback": "Try again after fixing the provider configuration.",
            },
            "expected_output": {
                "score": 0,
                "feedback": "Ask for root cause, minimal fix, and verification steps.",
            },
            "safety_against_bad_ai_fixes": {
                "score": 0,
                "feedback": "Ask the AI not to invent files or rewrite unrelated code.",
            },
        },
        "strengths": [],
        "weaknesses": [error_message],
        "improved_prompt": "Please restate the bug with the exact error, expected behavior, minimal-fix constraints, and a request for step-by-step debugging.",
        "would_this_likely_solve_the_bug": "No",
        "reason": "Automatic prompt evaluation failed, so this is a fallback response.",
    }


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
