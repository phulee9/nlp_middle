import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GroqLLM:

    def __init__(self, model_name: Optional[str] = None):

        from openai import OpenAI

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file:  GROQ_API_KEY=gsk_..."
            )

        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        # Groq exposes an OpenAI-compatible /v1 endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        logger.info(f"[GroqLLM] Initialized with model='{self.model_name}'")

    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"[GroqLLM] API call failed: {e}")
            raise