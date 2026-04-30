"""
groq_llm.py
───────────
GROQ API wrapper for LLM calls.

Groq provides blazing-fast inference for open-source models via their
custom LPU (Language Processing Unit) hardware.

Available models (as of 2025):
  - llama-3.1-8b-instant   – fast, good quality
  - llama-3.3-70b-versatile – better quality, slower
  - mixtral-8x7b-32768      – large context window (32k tokens)

We use the OpenAI-compatible endpoint so we can use the openai SDK.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GroqLLM:
    """
    Thin wrapper around the Groq API (OpenAI-compatible endpoint).

    Usage:
        llm = GroqLLM()
        answer = llm.generate(
            system_prompt="You are a helpful assistant.",
            user_message="What is RAG?",
        )
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialise the Groq client.

        Reads GROQ_API_KEY from environment (set in .env file).
        Reads GROQ_MODEL from environment, or uses the provided model_name.
        """
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
        """
        Call the Groq API and return the assistant's text response.

        Args:
            system_prompt: high-level instructions for the model
            user_message:  the actual user input
            temperature:   0.0 = deterministic, 1.0 = creative
            max_tokens:    maximum response length in tokens

        Returns:
            The model's text response as a plain string.
        """
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