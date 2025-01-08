import os
import re
from typing import Any, Literal

import backoff
import openai
from accelerate import Accelerator
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from config import EvaluateConfig
from evaluator import Evaluator


class WinRateEvaluator(Evaluator):
    def __init__(self, config: EvaluateConfig, accelerator: Accelerator) -> None:
        super().__init__(config, accelerator)

        api_key = os.environ.get("OPENAI_API_KEY", None)
        self.model = openai.Client(api_key=api_key)

        self.task: Literal["hh", "tldr"] = "hh"
        self.mode: Literal["concise", "simple", None] = None

    @backoff.on_exception(
        wait_gen=backoff.expo,
        exception=openai.APIConnectionError,
        logger="OpenAILanguageModel",
    )
    @staticmethod
    def completion_with_backoff(
        client: openai.Client,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 10,
    ) -> ChatCompletion:
        response = client.chat.completions.with_raw_response.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.parse()

    def _clean_response(self, response: str) -> str:
        match = re.search(r"[.!?](?=[^.!?]*$)", response)
        if match:
            return response[: match.start() + 1]
        return response

    def _create_prompt(self, history: str, first: str, second: str) -> str:
        """Creates prompt according to the template format.

        Args:
            first (str): first response, randomly shuffled between response + ref
            second (str): second response, randomly shuffled between response + ref

        Returns:
            str: templated prompt
        """
        if self.task == "hh":
            return (
                f"For the following query to a chatbot, which response is more helpful?\n\n"
                f"Query:\n{history}\n\n"
                f"Response A: {first}\n\n"
                f"Response B: {second}\n\n"
                "FIRST provide a one-sentence comparison of the two responses and explain "
                'which you feel is more helpful. SECOND, on a new line, state only "A" or "B" '
                "to indicate which response is more helpful. Your response should use the format:"
                "Comparison: <one-sentence comparison and explanation>\n"
                'More helpful: <"A" or "B">'
            )
        elif self.task == "tldr":
            if self.mode == "concise":
                return (
                    f"Which of the following summaries does a better job of summarizing the most "
                    "important points in the given forum post, without including unimportant or "
                    "irrelevant details? A good summary is both precise and concise.\n\n"
                    f"Post:\n{history}\n\n"
                    f"Summary A: {first}\n\n"
                    f"Summary B: {second}\n\n"
                    f'FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:\nComparison: <one-sentence comparison and explanation>\nPreferred: <"A" or "B">'
                )
            elif self.mode == "simple":
                return (
                    f"Which of the following summaries does a better job of summarizing the most "
                    "important points in the given forum post?\n\n"
                    f"Post:\n{history}\n\n"
                    f"Summary A: {first}\n\n"
                    f"Summary B: {second}\n\n"
                    f'FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:\nComparison: <one-sentence comparison and explanation>\nPreferred: <"A" or "B">'
                )

    def evaluate_sample(self, *args, **kwargs) -> dict[str, Any]:
        return super().evaluate_sample(*args, **kwargs)
