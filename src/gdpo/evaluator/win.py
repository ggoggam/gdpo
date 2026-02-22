import json
import os
import random
import re
from pathlib import Path

import backoff
import datasets
import numpy as np
import openai
from accelerate import Accelerator
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from tqdm import tqdm

from gdpo.config import EvaluateConfig
from gdpo.evaluator import Evaluator


class WinRateEvaluator(Evaluator):
    def __init__(self, config: EvaluateConfig, accelerator: Accelerator) -> None:
        super().__init__(config, accelerator)

        random.seed(config.seed)

        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.model = openai.Client(api_key=api_key)
        self.task = config.task
        self.mode = config.mode

        self.dataset = datasets.load_dataset(
            "json", data_files=config.output_path, split="train",
        ).shuffle(seed=config.seed)

    @backoff.on_exception(
        wait_gen=backoff.runtime,
        exception=openai.RateLimitError,
        value=lambda error: (
            float(
                error.response.headers.get(
                    "x-ratelimit-reset-requests", "1000",
                ).replace("ms", ""),
            )
            / 1000.0
        ),
        max_tries=100,
        logger="OpenAILanguageModel",
    )
    @backoff.on_exception(
        wait_gen=backoff.expo,
        exception=openai.APIConnectionError,
        logger="OpenAILanguageModel",
    )
    def completion_with_backoff(
        self,
        model_name: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float = 0.0,
        max_tokens: int = 10,
    ) -> ChatCompletion:
        response = self.model.chat.completions.with_raw_response.create(
            model=model_name,
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

    def _create_prompt(
        self, history: list[dict[str, str]], first: str, second: str,
    ) -> str:
        """Creates prompt according to the template format.

        Args:
            first (str): first response, randomly shuffled between response + ref
            second (str): second response, randomly shuffled between response + ref

        Returns:
            str: templated prompt

        """
        _tldr_format = (
            "FIRST provide a one-sentence comparison of the two summaries, explaining"
            ' which you prefer and why. SECOND, on a new line, state only "A" or "B"'
            " to indicate your choice. Your response should use the format:\n"
            "Comparison: <one-sentence comparison and explanation>\n"
            'Preferred: <"A" or "B">'
        )
        if self.task == "hh":
            history_text = ""
            for message in history[-1:]:
                history_text += f"{message['role'].upper()}: {message['content']}\n"
            return (
                "For the following query to a chatbot, which response is more"
                " helpful?\n\n"
                f"Query:\n{history_text}\n\n"
                f"Response A: {first}\n\n"
                f"Response B: {second}\n\n"
                "FIRST provide a one-sentence comparison of the two responses and"
                " explain which you feel is more helpful. SECOND, on a new line,"
                ' state only "A" or "B" to indicate which response is more helpful.'
                " Your response should use the format:"
                "Comparison: <one-sentence comparison and explanation>\n"
                'More helpful: <"A" or "B">'
            )
        if self.task == "tldr":
            history_text = history[0]["content"]
            if self.mode == "concise":
                return (
                    "Which of the following summaries does a better job of"
                    " summarizing the most important points in the given forum post,"
                    " without including unimportant or irrelevant details? A good"
                    " summary is both precise and concise.\n\n"
                    f"Post:\n{history_text}\n\n"
                    f"Summary A: {first}\n\n"
                    f"Summary B: {second}\n\n"
                    + _tldr_format
                )
            if self.mode == "simple":
                return (
                    "Which of the following summaries does a better job of"
                    " summarizing the most important points in the given forum"
                    " post?\n\n"
                    f"Post:\n{history_text}\n\n"
                    f"Summary A: {first}\n\n"
                    f"Summary B: {second}\n\n"
                    + _tldr_format
                )
        return None

    def evaluate(self) -> dict[str, any]:
        pbar = tqdm(total=self.config.samples)
        num_samples = 0

        file_path = Path(self.config.output_path).parent / "compare.jsonl"

        response_wins = []
        _min = _max = _mean = _stde = 0.0

        with file_path.open("w") as file:
            for sample in self.dataset:
                if num_samples >= self.config.samples:
                    break

                comparison_result = None
                # choose one reference response
                # if there are multiple, choose one randomly
                if isinstance(sample["chosen"], str):
                    reference_response = sample["chosen"]
                else:
                    if len(sample["chosen"]) < 1:
                        continue
                    reference_response = random.choice(sample["chosen"])

                # shuffle generated outputs
                outputs = list(sample["outputs"])
                random.shuffle(outputs)

                completions = []
                for output in outputs[: self.config.n]:
                    # NOTE: shuffle generated response vs reference response
                    # due to order bias of language models
                    shuffled = [(0, output), (1, reference_response)]
                    random.shuffle(shuffled)
                    prompt = self._create_prompt(
                        sample["history"], shuffled[0][1], shuffled[1][1],
                    )
                    completion = self.completion_with_backoff(
                        model_name=self.config.model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=self.config.max_tokens,
                    )
                    completions.append(completion.choices[0].message.content)

                if self.config.task == "tldr":
                    answers = [
                        re.search(r"(?i)(preferred): (.).*", completion)
                        for completion in completions
                    ]
                else:
                    answers = [
                        re.search(r"(?i)(more helpful): (.).*", completion)
                        for completion in completions
                    ]

                # skip if any of the completions did not match the regex
                if any(answer is None for answer in answers):
                    continue
                answers = [answer.group(2).lower() for answer in answers]

                votes = []
                for answer in answers:
                    if answer == "a":
                        votes.append(shuffled[0][0] == 0)
                    elif answer == "b":
                        votes.append(shuffled[1][0] == 0)
                if len(votes) == self.config.n:
                    response_wins.append(votes)
                    comparison_result = {
                        "reference_response": reference_response,
                        "wins": votes,
                    }

                if comparison_result:
                    file.write(
                        json.dumps(
                            {
                                "generated": sample,
                                "reference": comparison_result,
                            },
                        )
                        + "\n",
                    )
                    num_samples += 1
                    pbar.update(1)

                    aggregated = np.array(response_wins).mean(axis=0)
                    _min = aggregated.min()
                    _max = aggregated.max()
                    _mean = aggregated.mean()
                    _stde = aggregated.std() / np.sqrt(len(aggregated))
                    pbar.set_description(
                        f"response vs reference wins: {_mean:.3f} ± {_stde},"
                        f" min: {_min:.3f}, max: {_max:.3f}",
                    )

        pbar.close()

        return {"mean": _mean, "std_err": _stde, "min": _min, "max": _max}
