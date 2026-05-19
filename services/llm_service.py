from typing import Any

from utils.helpers import get_secret


class LLMService:
    GEMINI_MODEL = "gemini-1.5-flash"
    GROQ_MODEL = "llama-3.1-8b-instant"
    COHERE_MODEL = "command-r"
    OPENAI_MODEL = "gpt-4o-mini"

    @staticmethod
    def configured_providers(api_key_overrides: dict[str, str] | None = None) -> list[str]:
        providers = []
        for provider in ["Gemini", "Groq", "OpenAI", "Cohere"]:
            if LLMService.provider_has_key(provider, api_key_overrides):
                providers.append(provider)
        return providers

    @staticmethod
    def provider_has_key(
        provider: str,
        api_key_overrides: dict[str, str] | None = None,
    ) -> bool:
        api_key_overrides = api_key_overrides or {}

        if provider == "Gemini":
            return bool(api_key_overrides.get("Gemini") or get_secret("GEMINI_API_KEY"))
        if provider == "Groq":
            return bool(api_key_overrides.get("Groq") or get_secret("GROQ_API_KEY"))
        if provider == "OpenAI":
            return bool(api_key_overrides.get("OpenAI") or get_secret("OPENAI_API_KEY"))
        if provider == "Cohere":
            return bool(api_key_overrides.get("Cohere") or get_secret("COHERE_API_KEY"))
        return False

    @staticmethod
    def get_provider_details(
        provider: str,
        api_key_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if provider == "Gemini":
            return {
                "provider": "Gemini",
                "model": LLMService.GEMINI_MODEL,
                "configured": LLMService.provider_has_key("Gemini", api_key_overrides),
            }
        if provider == "Groq":
            return {
                "provider": "Groq",
                "model": LLMService.GROQ_MODEL,
                "configured": LLMService.provider_has_key("Groq", api_key_overrides),
            }
        if provider == "OpenAI":
            return {
                "provider": "OpenAI",
                "model": LLMService.OPENAI_MODEL,
                "configured": LLMService.provider_has_key("OpenAI", api_key_overrides),
            }
        if provider == "Cohere":
            return {
                "provider": "Cohere",
                "model": LLMService.COHERE_MODEL,
                "configured": LLMService.provider_has_key("Cohere", api_key_overrides),
            }
        if provider == "Auto Balanced":
            providers = LLMService.configured_providers(api_key_overrides)
            return {
                "provider": "Auto Balanced",
                "model": ", ".join(providers) if providers else "None configured",
                "configured": bool(providers),
            }
        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def generate(
        provider: str,
        prompt: str,
        api_key_overrides: dict[str, str] | None = None,
    ) -> str:
        api_key_overrides = api_key_overrides or {}

        if provider == "Gemini":
            return LLMService._generate_with_gemini(prompt, api_key_overrides.get("Gemini"))
        if provider == "Groq":
            return LLMService._generate_with_groq(prompt, api_key_overrides.get("Groq"))
        if provider == "OpenAI":
            return LLMService._generate_with_openai(prompt, api_key_overrides.get("OpenAI"))
        if provider == "Cohere":
            return LLMService._generate_with_cohere(prompt, api_key_overrides.get("Cohere"))
        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _generate_with_gemini(prompt: str, api_key_override: str | None = None) -> str:
        import google.generativeai as genai

        api_key = api_key_override or get_secret("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(LLMService.GEMINI_MODEL)
        response = model.generate_content(prompt)
        return getattr(response, "text", "") or ""

    @staticmethod
    def _generate_with_groq(prompt: str, api_key_override: str | None = None) -> str:
        from groq import Groq

        api_key = api_key_override or get_secret("GROQ_API_KEY")
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
    def _generate_with_openai(prompt: str, api_key_override: str | None = None) -> str:
        from openai import OpenAI

        api_key = api_key_override or get_secret("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=LLMService.OPENAI_MODEL,
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
    def _generate_with_cohere(prompt: str, api_key_override: str | None = None) -> str:
        import cohere

        api_key = api_key_override or get_secret("COHERE_API_KEY")
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
