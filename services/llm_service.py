from typing import Any

import cohere
import google.generativeai as genai
from groq import Groq

from utils.helpers import get_secret


class LLMService:
    GEMINI_MODEL = "gemini-1.5-flash"
    GROQ_MODEL = "llama-3.1-8b-instant"
    COHERE_MODEL = "command-r"

    @staticmethod
    def configured_providers() -> list[str]:
        providers = []
        if get_secret("GEMINI_API_KEY"):
            providers.append("Gemini")
        if get_secret("GROQ_API_KEY"):
            providers.append("Groq")
        if get_secret("COHERE_API_KEY"):
            providers.append("Cohere")
        return providers

    @staticmethod
    def provider_has_key(provider: str) -> bool:
        if provider == "Gemini":
            return bool(get_secret("GEMINI_API_KEY"))
        if provider == "Groq":
            return bool(get_secret("GROQ_API_KEY"))
        if provider == "Cohere":
            return bool(get_secret("COHERE_API_KEY"))
        return False

    @staticmethod
    def get_provider_details(provider: str) -> dict[str, Any]:
        if provider == "Gemini":
            return {
                "provider": provider,
                "model": LLMService.GEMINI_MODEL,
                "configured": LLMService.provider_has_key(provider),
            }
        return {
            "provider": "Groq",
                "model": LLMService.GROQ_MODEL,
                "configured": LLMService.provider_has_key("Groq"),
            }
        if provider == "Cohere":
            return {
                "provider": "Cohere",
                "model": LLMService.COHERE_MODEL,
                "configured": LLMService.provider_has_key("Cohere"),
            }
        if provider == "Auto Balanced":
            providers = LLMService.configured_providers()
            return {
                "provider": "Auto Balanced",
                "model": ", ".join(providers) if providers else "None configured",
                "configured": bool(providers),
            }
        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def generate(provider: str, prompt: str) -> str:
        if provider == "Gemini":
            return LLMService._generate_with_gemini(prompt)
        if provider == "Groq":
            return LLMService._generate_with_groq(prompt)
        if provider == "Cohere":
            return LLMService._generate_with_cohere(prompt)
        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _generate_with_gemini(prompt: str) -> str:
        api_key = get_secret("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(LLMService.GEMINI_MODEL)
        response = model.generate_content(prompt)
        return getattr(response, "text", "") or ""

    @staticmethod
    def _generate_with_groq(prompt: str) -> str:
        api_key = get_secret("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=LLMService.GROQ_MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You return only valid JSON and never wrap it in markdown.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _generate_with_cohere(prompt: str) -> str:
        api_key = get_secret("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY is not configured.")

        client = cohere.ClientV2(api_key=api_key)
        response = client.chat(
            model=LLMService.COHERE_MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You return only valid JSON and never wrap it in markdown.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        text = ""
        for item in getattr(response.message, "content", []) or []:
            if getattr(item, "type", "") == "text":
                text += getattr(item, "text", "")
        return text
