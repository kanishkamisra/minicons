"""Utilities for the OpenAI API to access models like GPT-3.5 and 4."""

import re
import torch
import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .utils import between

def register_api_key(path):
    """Register the API key for the OpenAI API."""
    openai.api_key_path = path

class ResponseObj:
    """A class to hold the response from the OpenAI API."""

    def __init__(self, response):
        self.tokens = response["tokens"]
        self.logprobs = torch.tensor(
            response["token_logprobs"][1:], dtype=torch.float32
        )
        self.offsets = response["text_offset"]
        self.sequence = "".join(self.tokens)

    def sequence_score(
        self, surp=False, bits=False, reduction=lambda x: x.mean().item()
    ):
        """Get the sequence score."""
        scores = self.logprobs

        if surp:
            scores = -1.0 * scores
        if bits:
            scores = scores / torch.tensor(2.0).log()

        return reduction(scores)

    def conditional_score(
        self, phrase, surp=False, bits=False, normalize=True
    ):
        """Get the conditional scores for a phrase contained in the sequence."""
        phrase = (
            f" {phrase.strip()}"
            if phrase.strip()[0] not in ",."
            else phrase.strip()
        )

        # TODO make this a param. for now, just take the last instance.
        target_span = list(re.finditer(phrase, self.sequence))[-1].span()

        mask = torch.tensor(
            [int(between(i, target_span)) for i in self.offsets[1:]],
            dtype=torch.float32,
        )
        length = mask.sum()

        scores = mask * self.logprobs

        if surp:
            scores = -1.0 * scores
        if bits:
            scores = scores / torch.tensor(2.0).log()

        if normalize:
            scores = scores / length

        return scores.sum().item()

    def token_score(self, surp=False, bits=False, prob=False):
        """Get the token logprob/surprisal for each token in the sequence."""
        scores = self.logprobs

        # make sure surp and prob are both not True
        assert not (surp and prob)

        # make sure bits and prob are both not True
        assert not (bits and prob)

        if surp:
            scores = -1.0 * scores
        if bits:
            scores = scores / torch.tensor(2.0).log()
        if prob:
            scores = scores.exp()

        scores = scores.tolist()
        token_scores = list(zip(self.tokens[1:], scores))

        return token_scores


class OpenAIQuery:
    """A class to query the OpenAI API."""

    def __init__(self, model, prompts):
        self.prompts = prompts
        self.model = model

    @retry(
        wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)
    )
    def query(self):
        """Query the OpenAI API."""
        self.response = openai.Completion.create(
            engine=self.model,
            prompt=self.prompts,
            max_tokens=0,
            temperature=0.0,
            logprobs=0,
            echo=True,
        )

    def sequence_score(
        self, surp=False, bits=False, reduction=lambda x: x.mean().item()
    ):
        """Get the sequence score for each response."""
        scores = []
        for resp in self.response["choices"]:
            resp_obj = ResponseObj(resp["logprobs"])
            scores.append(resp_obj.sequence_score(surp, bits, reduction))

        return scores

    def conditional_score(
        self, phrases, surp=False, bits=False, normalize=True
    ):
        """Get the conditional score for each response."""
        assert len(phrases) == len(self.response["choices"])

        scores = []
        for resp, phrase in zip(self.response["choices"], phrases):
            resp_obj = ResponseObj(resp["logprobs"])
            scores.append(
                resp_obj.conditional_score(phrase, surp, bits, normalize)
            )

        return scores

    def token_score(self, surp=False, bits=False, prob=False):
        """Get the token score for each response."""
        scores = []
        for resp in self.response["choices"]:
            resp_obj = ResponseObj(resp["logprobs"])
            scores.append(resp_obj.token_score(surp, bits, prob))

        return scores
