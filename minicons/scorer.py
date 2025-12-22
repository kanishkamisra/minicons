"""Utilities for scoring sequences using Language Models."""

from typing import (
    Iterable,
    Union,
    List,
    Collection,
    Optional,
    Callable,
    Tuple,
    Any,
    cast,
)

import re
import torch
import warnings

from collections import defaultdict
from itertools import chain

import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BatchEncoding,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText
)

from PIL import Image

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except:
    pass

from transformers.utils.logging import set_verbosity_error

from .utils import batch_wise_logprobs, all_equal

set_verbosity_error()


class LMScorer:
    """
    Base LM scorer class intended to store models and tokenizers along
    with methods to facilitate the analysis of language model output scores.
    """

    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        device: Optional[str] = "cpu",
        tokenizer=None,
        **kwargs,
    ) -> None:
        """
        :param model: should be path to a model (.pt or .bin file) stored
            locally, or name of a pretrained model stored on the Huggingface
            Model Hub, or a model (torch.nn.Module) that have the same
            signature as the corresponding Huggingface model (see the subclass
            for details).
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        :param tokenizer: if provided, use this tokenizer.
        """
        if tokenizer is not None:
            if isinstance(tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
            else:
                self.tokenizer = tokenizer
        elif isinstance(model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        else:
            raise Exception("Must provide either model name or tokenizer.")
        self.device = device
        self.vocab = defaultdict(list)
        # {self.vocab[x.strip()].append(i) for x, i in [(self.tokenizer.decode([i]), i) for i in range(self.tokenizer.vocab_size)]}
        for i in range(self.tokenizer.vocab_size):
            decoded = [(self.tokenizer.decode(i), i)]
            for x, j in decoded:
                self.vocab[x.strip()].append(j)

    def add_special_tokens(self, text: Union[str, Iterable[str]]) -> List[str]:
        raise NotImplementedError

    def distribution(self, batch: Iterable) -> torch.Tensor:
        raise NotImplementedError

    def topk(self, distribution: torch.Tensor, k: int = 1) -> Tuple:
        top_k = distribution.topk(k)

        probs = top_k.values.squeeze(1).exp().tolist()
        if k == 1:
            tokens = self.decode(top_k.indices.squeeze(1))
        else:
            tokens = [self.decode(x) for x in top_k.indices.squeeze(1)]

        return tokens, probs

    # def query(self, distribution: torch.Tensor, queries: List[str]) -> Tuple:
    #     # this will be self.vocab tho
    #     query_ids = [self.vocab[a] for a in queries]
    #     maxlen = max(map(len, query_ids))
    #     query_ids = [
    #         (
    #             q + [self.tokenizer.pad_token_id] * (maxlen - len(q))
    #             if len(q) < maxlen
    #             else q
    #         )
    #         for q in query_ids
    #     ]
    #     current_batch_size = distribution.shape[0]
    #     probs = (
    #         distribution[torch.arange(current_batch_size)[:, None], query_ids]
    #         .max(1)
    #         .values.exp()
    #         .tolist()
    #     )

    #     inv_ranks = distribution.argsort().argsort() + 1
    #     ranks = distribution.shape[1] - inv_ranks + 1
    #     token_ranks = (
    #         ranks[torch.arange(current_batch_size)[:, None], query_ids]
    #         .min(1)
    #         .values.tolist()
    #     )

    #     return probs, token_ranks

    def query(
        self,
        distribution: torch.Tensor,
        queries: List[List[str]],
        prob: bool = True,
        tolist: bool = True,
    ):
        """Queries distributions for (log)probabilities of target tokens."""
        if isinstance(queries[0], str):
            queries = [queries]

        scores = []
        token_ranks = []
        for i, querylist in enumerate(queries):
            query_ids = [self.vocab[a] for a in querylist]
            maxlen = max(map(len, query_ids))
            query_ids = [
                (q + [self.pad_token_id] * (maxlen - len(q)) if len(q) < maxlen else q)
                for q in query_ids
            ]
            query_logprobs = distribution[i, query_ids].max(1).values

            inv_ranks = distribution.argsort().argsort() + 1
            ranks = distribution.shape[1] - inv_ranks + 1
            query_token_ranks = ranks[i, query_ids].min(1).values.tolist()

            scores.append(query_logprobs)
            token_ranks.append(query_token_ranks)

        if prob:
            scores = [s.exp() for s in scores]

        if tolist:
            scores = [s.tolist() for s in scores]

        return scores, token_ranks

    def logprobs(
        self, batch: Iterable, rank: bool = False
    ) -> Union[float, List[float]]:
        warnings.warn(
            "logprobs is deprecated, use compute_stats instead", DeprecationWarning
        )
        raise NotImplementedError

    def compute_stats(self, batch: Iterable, rank: bool = False) -> Union[
        Tuple[List[float], List[int]],
        List[float],
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        raise NotImplementedError

    def prepare_text(self, text: Union[str, List[str], BatchEncoding]) -> Any:
        raise NotImplementedError

    def prime_text(
        self,
        prefix: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        separator: str = " ",
    ) -> Tuple:
        raise NotImplementedError

    def token_score(
        self,
        batch: Union[str, List[str]],
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        rank: bool = False,
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        """
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        """
        raise NotImplementedError

    def word_score(
        self,
        batch: Union[str, List[str]],
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        rank: bool = False,
        bow_correction: bool = False,
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        """
        Wraps token_score's outputs into word-level metrics:
            `(word, score)`,
        where score represents the log-probability (by default) of the word given context.
        Token probabilities are summed across the whole word. Words are currently split on spaces and punctuation.

        Args are the same as token_score except for `agg_method`
        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)

        Outputs are in the same format as token_score outputs
        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        """
        all_token_scores = self.token_score(
            batch=batch,
            surprisal=surprisal,
            prob=prob,
            base_two=base_two,
            rank=rank,
            bow_correction=bow_correction,
        )
        all_word_scores = []
        for i in range(len(all_token_scores)):
            if type(batch) == str:
                sentence = batch
            else:
                sentence = batch[i]
            words = re.findall(r"[\w']+|[.,!?;]", sentence)
            token_scores = all_token_scores[i]
            # if token_score pads the beginning with a token (i.e. like llama)
            if token_scores[0][0] == self.tokenizer.special_tokens_map["bos_token"]:
                token_scores = token_scores[1:]
            token_index = 0
            word_index = 0
            word_scores = []  # list of word, surprisal tuples
            try:
                while token_index < len(token_scores):
                    current_word = words[word_index]
                    current_token, current_surprisal = token_scores[token_index]
                    # token does not match, alignment must be adjusted
                    mismatch = current_token != current_word
                    while mismatch:
                        token_index += 1
                        current_token += token_scores[token_index][0]
                        current_surprisal += token_scores[token_index][1]
                        mismatch = current_token != current_word
                    word_scores.append((current_word, current_surprisal))
                    token_index += 1
                    word_index += 1
            except Exception:
                warning_message = f"Failed to aggregate word-level scores for {sentence}, returning token-level scores"
                warnings.warn(warning_message)
                word_scores = token_scores
            all_word_scores.append(word_scores)
        return all_word_scores

    def score(
        self, batch: Union[str, List[str]], pool: Callable = torch.mean, *args
    ) -> Union[float, List[float]]:
        """
        DEPRECATED as of v 0.1.18. Check out ``sequence_score`` or ``token_score`` instead!

        Pooled estimates of sentence log probabilities, computed by the
        language model. Pooling is usually done using a function that
        is passed to the method.

        :param batch: a list of sentences that will be passed to the
            language model to score.
        :type batch: Union[str, List[str]]
        :param pool: Pooling function, is selected to be
            `torch.mean()` by default.
        :type pool: Callable

        :return: Float or list of floats specifying the log
            probabilities of the input sentence(s).
        :rtype: Union[float, List[float]]
        """
        warnings.warn(
            "score is deprecated, use sequence_score or token_score instead",
            DeprecationWarning,
        )

    def adapt_score(
        self,
        preamble: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        pool: Callable = torch.mean,
        *args,
    ) -> None:
        """
        DEPRECATED as of v 0.2.10. Check out ``partial_score`` instead!
        """
        warnings.warn(
            "adapt_score is deprecated, use conditional_score or token_score instead",
            DeprecationWarning,
        )

    def partial_score(
        self,
        preamble: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        separator: str = " ",
        reduction: Callable = lambda x: x.mean(0).item(),
        **kwargs,
    ) -> List[float]:
        warnings.warn(
            "partial_score is deprecated, use conditional_score instead",
            DeprecationWarning,
        )

    def conditional_score(
        self,
        prefix: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        separator: str = " ",
        reduction: Callable = lambda x: x.mean(0).item(),
        prob: bool = False,
        base_two: bool = False,
        bow_correction: bool = False,
        **kw,
    ) -> List[float]:
        """
        Pooled estimates of sequence log probabilities (or some modification of it), given a prefix. Pooling is usually done using a function that is passed to the method.

        :param prefix: a batch of prefixes or primes passed to the
            language model. This is what the sequence is conditioned on, and the model ignores the word probabilities of this part of the input in estimating the overall score.
        :type prefix: ``Union[str, List[str]]``
        :param stimuli: a batch of sequences (same length as prefix)
            that form the main input consisting of the sequence whose
            score you want to calculate.
        :type stimuli: ``Union[str, List[str]]``
        :param reduction: Reduction function, is selected to be
            ``lambda x: x.mean(0).item()`` by default, which stands for the avg. log-probability per token for each sequence in the batch.
        :type reduction: Callable
        :param kw: model-specific keyword arguments to pass to the `prepare_text` function
        :return: List of floats specifying the desired score for the stimuli part of the input, e.g., P(stimuli | preamble).
        :rtype: ``List[float]``
        """
        primed = self.prime_text(prefix, stimuli, separator, **kw)

        result = self.compute_stats(
            primed,
            rank=False,
            base_two=base_two,
            prob=prob,
            bow_correction=bow_correction,
            return_tensors=True,
        )
        logprob = result
        reduced = list(map(reduction, logprob))

        return reduced

    def sequence_score(
        self,
        batch,
        reduction=lambda x: x.mean(0).item(),
        prob: bool = False,
        base_two: bool = False,
        bow_correction: bool = False,
        **kw,
    ):
        """
        Pooled estimates of sequence log probabilities (or some modification of it).

        :param batch: a batch of sequences whose score you want to calculate.
        :type batch: ``Union[str, List[str]]``
        :param reduction: Reduction function, is selected to be
            ``lambda x: x.mean(0).item()`` by default, which stands for the avg. log-probability per token for each sequence in the batch.
        :type reduction: Callable
        :param kw: model-specific keyword arguments to pass to the `prepare_text` function
        :return: List of floats specifying the desired score for the stimuli part of the input, e.g., P(stimuli | preamble).
        :rtype: ``List[float]``

        TODO: reduction should be a string, if it's a function, specify what kind of function. --> how to ensure it is always that type?
        """
        tokenized = self.prepare_text(batch, **kw)
        scores = self.compute_stats(
            tokenized,
            rank=False,
            base_two=base_two,
            prob=prob,
            bow_correction=bow_correction,
            return_tensors=True,
        )
        reduced = list(map(reduction, scores))
        return reduced

    def encode(
        self,
        text: Union[str, List[str]],
        manual_special: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> BatchEncoding:
        """
        Encode a batch of sentences using the model's tokenizer.
        Equivalent of calling `model.tokenizer(input)`

        :param ``Union[str, List[str]]`` text: Input batch/sentence to
            be encoded.
        :param manual_special: Specification of whether special tokens
            will be manually encoded.
        :type manual_special: bool
        :param return_tensors: returned tensor format. Default `'pt'`
        :type manual_special: str

        :return: Encoded batch
        :rtype: ``BatchEncoding``
        """
        sentences = [text] if isinstance(text, str) else text

        if manual_special:
            # manually add special tokens
            sentences = self.add_special_tokens(sentences)
            if return_tensors:
                tokens = self.tokenizer.batch_encode_plus(
                    sentences,
                    add_special_tokens=False,
                    padding="longest",
                    return_attention_mask=True,
                    return_tensors=return_tensors,
                )
        else:
            # mostly for masked LMs
            tokens = self.tokenizer.batch_encode_plus(
                sentences, padding="longest", return_attention_mask=True
            )

        return tokens

    def decode(self, idx: List[int]):
        """
        Decode input ids using the model's tokenizer.

        :param ``List[int]`` idx: List of ids.

        :return: Decoded strings
        :rtype: List[str]
        """
        return [
            self.tokenizer.decode([x]).strip()
            for x in self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.convert_ids_to_tokens(idx)
            )
        ]


class MaskedLMScorer(LMScorer):
    """
    Class for Masked Langauge Models such as BERT, RoBERTa, etc.

    :param model: should be path to a model (.pt or .bin file) stored locally,
        or name of a pretrained model stored on the Huggingface Model Hub, or
        a model (torch.nn.Module) that have the same signature as a
        Huggingface model obtained from `AutoModelForMaskedLM`. In the last
        case, a corresponding tokenizer must also be provided.
    :param device: device type that the model should be loaded on,
        options: `cpu or cuda:{0, 1, ...} or auto`
    :type device: str, optional
    :param tokenizer: if provided, use this tokenizer.
    """

    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        device: Optional[str] = "cpu",
        tokenizer=None,
        PLL_metric: str = "original",
        **kwargs,
    ) -> None:
        """
        :param model: should be path to a model (.pt or .bin file) stored
            locally, or name of a pretrained model stored on the Huggingface
            Model Hub, or a model (torch.nn.Module) that have the same
            signature as a Huggingface model obtained from
            `AutoModelForMaskedLM`. In the last case, a corresponding tokenizer
            must also be provided.
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        :param tokenizer: if provided, use this tokenizer.
        """
        super(MaskedLMScorer, self).__init__(model, device=device, tokenizer=tokenizer)

        if isinstance(model, str):
            if self.device == "auto":
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model, device_map=self.device, return_dict=True, **kwargs
                )
            else:
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model, return_dict=True, **kwargs
                )
            # self.model.to(self.device)
        else:
            self.model = model

        if self.device != "auto":
            self.model.to(self.device)
        self.model.eval()

        self.PLL_metric: str = PLL_metric
        # define CLS and SEP tokens
        self.bos_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def add_special_tokens(self, text: Union[str, Iterable[str]]) -> List[str]:
        """
        Reformats input text to add special model-dependent tokens.

        :param text: single string or batch of strings to be
            modified.
        :type text: ``Union[str, List[str]]``

        :return: Modified input, containing special tokens as per
            tokenizer specification
        :rtype: ``List[str]``
        """
        sentences = [text] if isinstance(text, str) else text
        sentences = [
            self.tokenizer.cls_token + " " + sentence + " " + self.tokenizer.sep_token
            for sentence in sentences
        ]

        return sentences

    def mask(
        self, sentence_words: Union[Tuple[str, str], List[Tuple[str, str]]]
    ) -> Tuple[List[str], List[str]]:
        """
        Processes a list of (sentence, word) into input that has the
        word masked out of the sentence.

        Note: only works for masked LMs.

        :param ``Union[Tuple[str], List[Tuple[str]]]`` sentence_words:
            Input consisting of `[(sentence, word)]`, where sentence
            is an input sentence, and word is a word present in the
            sentence that will be masked out.

        :return: Tuple `(sentence, word, length)`
        """
        sentence_words = (
            [sentence_words] if isinstance(sentence_words, tuple) else sentence_words
        )
        sentences: List[str] = []
        words: List[str] = []

        for sentence, word in sentence_words:
            words.append(word)
            sentences.append(
                re.sub(
                    rf"(?<![\w\/-])({word})(?=[^\w\/-])",
                    self.tokenizer.mask_token,
                    sentence,
                )
            )

        return (sentences, words)

    def cloze(
        self,
        sentence_words: Union[Tuple[str, str], List[Tuple[str, str]]],
        PLL_metric: Optional[str] = None,
        probs: Optional[bool] = False,
    ) -> List[float]:
        """
        Runs inference on masked input.
        Note: only works for masked LMs.

        :param ``Union[Tuple[str], List[Tuple[str]]]`` sentence_words:
            Input consisting of `[(sentence, word)]`, where sentence
            is an input sentence, and word is a word present in the
            sentence that will be masked out and inferred.
        :param PLL_metric: PLL scoring strategy to be used.
            Options: `original` or `within_word_l2r`. Default: `original`
            For motivation as to why to use `within_word_l2r` PLL scoring, see Kauf & Ivanova (2023):
            https://arxiv.org/abs/2305.10588
        :param probs: whether to return probabilities (if True) or log probabilities (if False)

        :return: A list of tensors corresponding to (log) probabilities for the desired word
            in context
        """
        sentences = list(map(lambda x: x[0], sentence_words))
        encoded = self.tokenizer(sentences, return_tensors="pt", padding=True)
        targets_start = []
        targets_end = []

        # Iterating over sentence-target word pairs
        for batch_index, (sentence, word) in enumerate(sentence_words):
            desired_tokens = self.tokenizer(
                word, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            if PLL_metric == "within_word_l2r":
                start_idx = None
                word_ids = encoded.word_ids(batch_index=batch_index)
                # Iterating over all words in the sentence
                for word_id in set(word_ids):
                    # Ignoring special tokens
                    if word_id is None:
                        continue
                    # Finding all tokens corresponding to the chosen word
                    indices = np.where(list(map(lambda x: x == word_id, word_ids)))[0]
                    tokens = encoded["input_ids"][batch_index][indices]
                    # Checking if the chosen word matches the target word
                    if torch.equal(tokens, desired_tokens):
                        start_idx = indices[0]
                if start_idx:
                    targets_start.append(start_idx)
                    targets_end.append(start_idx + len(desired_tokens))
                else:
                    raise ValueError(
                        f"Word ``{word}'' not found in sentence ``{sentence}''. PLL=within_word_l2r won't work if ``{word}'' is a subword or multiple words."
                    )
            else:
                for start_idx in range(
                    len(encoded["input_ids"][batch_index]) - len(desired_tokens)
                ):
                    # Checking if the chosen sequence of tokens matches the target sequence of tokens
                    if torch.equal(
                        encoded["input_ids"][batch_index][
                            start_idx : len(desired_tokens) + start_idx
                        ],
                        desired_tokens,
                    ):
                        targets_start.append(start_idx)
                        targets_end.append(start_idx + len(desired_tokens))

        if self.device != "auto":
            encoded = encoded.to(self.device)

        masked_tensors = self.get_masked_tensors(
            encoded,
            PLL_metric=PLL_metric,
            targets_start=targets_start,
            targets_end=targets_end,
        )

        target_prob_list = []

        with torch.no_grad():
            for masked_tensor, attn_mask, token_ids, token_indices in masked_tensors:
                masked_logits = (
                    self.model(input_ids=masked_tensor, attention_mask=attn_mask)
                    .logits[torch.arange(len(token_indices)), token_indices]
                    .squeeze()
                    .detach()
                )

                if len(token_indices) > 1:
                    logprobs = masked_logits - masked_logits.logsumexp(1).unsqueeze(1)
                    target_prob = (
                        logprobs[torch.arange(len(token_indices)), token_ids]
                        .squeeze()
                        .sum()
                    )
                else:
                    logprobs = masked_logits - masked_logits.logsumexp(0)
                    target_prob = logprobs[token_ids].squeeze().sum()

                target_prob_list.append(target_prob)

        target_probs_tensor = torch.tensor(target_prob_list)
        if probs:
            target_probs_tensor = target_probs_tensor.exp()

        return target_probs_tensor.tolist()

    def get_masked_tensors(
        self,
        encoded: BatchEncoding,
        PLL_metric: Optional[str] = None,
        targets_start: Optional[List[int]] = None,
        targets_end: Optional[List[int]] = None,
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Create batches of tokenized instances with each token in the sequence replaced by the mask token."""
        if PLL_metric is None:
            PLL_metric = self.PLL_metric

        for batch_index, (token_ids, attention_mask) in enumerate(
            zip(encoded["input_ids"], encoded["attention_mask"])
        ):
            token_ids = torch.as_tensor(token_ids)
            attention_mask = torch.as_tensor(attention_mask)

            target_token_indices: List[int] = []
            target_token_ids: List[torch.Tensor] = []

            # select tokens (and their indices) that will be predicted
            for token_index, token_id in enumerate(token_ids):
                if (
                    token_id != self.pad_token_id
                    and token_id != self.cls_token_id
                    and token_id != self.sep_token_id
                    and (
                        targets_start is None
                        or token_index >= targets_start[batch_index]
                    )
                    and (targets_end is None or token_index < targets_end[batch_index])
                ):
                    target_token_ids.append(token_id)
                    target_token_indices.append(token_index)

            target_token_indices = list(target_token_indices)
            target_token_ids = list(target_token_ids)

            # mask tokens based on the current token to be predicted
            mask_indices: List[List[int]]

            if PLL_metric == "within_word_l2r":
                """
                Future tokens belonging to the same word as the target token are masked during token inference as well.
                """
                word_ids = encoded.word_ids(
                    batch_index=batch_index
                )  # only used for this PLL_metric

                mask_indices = [
                    # mask the target token and all following tokens which belong to the same word
                    (
                        [mask_pos]
                        + [
                            j
                            for j in range(mask_pos + 1, target_token_indices[-1] + 1)
                            if word_ids[j] == word_ids[mask_pos]
                        ]
                        if word_ids[mask_pos] is not None
                        else [mask_pos]
                    )  # mask this token
                    for mask_pos in target_token_indices
                ]

            elif PLL_metric == "original":
                # Original PLL metric
                mask_indices = [[target] for target in target_token_indices]

            else:
                raise ValueError(f"PLL metric '{PLL_metric}' not supported.")

            # repeat the token ids and mask each set of tokens in a separate row
            token_ids_masked = token_ids.repeat(len(target_token_indices), 1)

            for i, mask_set in enumerate(mask_indices):
                token_ids_masked[i, mask_set] = self.mask_token_id

            yield (
                # token ids with some replaced by the mask token (effective tokens are replaced, but potentially more)
                token_ids_masked,
                # the attention mask is identical for all masked sets
                attention_mask.expand(len(target_token_indices), -1),
                # ids of the tokens to be predicted
                torch.tensor(target_token_ids),
                # indices of the tokens to be predicted
                torch.tensor(target_token_indices),
            )

    def prepare_text(
        self,
        text: Union[str, List[str], BatchEncoding],
        PLL_metric: Optional[str] = "original",
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Prepares a batch of input text into a format fit to run MLM
        scoring on.

        Borrows preprocessing algorithm from Salazar et al. (2020), and
        modifies code from the following github repository by simonpri:
        https://github.com/simonepri/lm-scorer

        :param text: batch of sentences to be prepared for scoring.
        :param PLL_metric: PLL scoring strategy to be used.
            Options: `original` or `within_word_l2r`. Default: `original`
            For motivation as to why to use `within_word_l2r` PLL scoring, see Kauf & Ivanova (2023):
            https://arxiv.org/abs/2305.10588

        :return: Batch of formatted input that can be passed to `logprob`
        """
        # converts input text to batch of tensors with every position except the cls and sep token masked

        if isinstance(text, BatchEncoding):
            encoded = text
        else:
            sentences = [text] if isinstance(text, str) else text
            encoded = self.encode(sentences, manual_special=False)

        return self.get_masked_tensors(encoded, PLL_metric)

    def prime_text(
        self,
        prefix: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        suffix: Union[None, str, List[str]] = None,
        separator: str = " ",
        PLL_metric: Optional[str] = None,
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on.

        Borrows preprocessing algorithm from Salazar et al. (2020), and
        modifies code from the following github repository by simonpri:
        https://github.com/simonepri/lm-scorer

        :param ``Union[str, List[str]]`` preamble: Batch of prefixes/prime/preambles on which the LM is conditioned.
        :param ``Union[str, List[str]]`` stimuli: Batch of continuations that are scored based on the conditioned text (provided in the ``preamble``). The positions of the elements match their counterparts in the ``preamble``.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        if isinstance(stimuli, str):
            assert isinstance(prefix, str)
            prefix = [prefix]
            stimuli = [stimuli]

            if suffix is not None:
                assert isinstance(suffix, str)

        assert isinstance(prefix, list)
        assert suffix is None or isinstance(suffix, list)

        assert len(prefix) == len(stimuli)
        assert suffix is None or len(suffix) == len(stimuli)

        # compute the length of each preamble
        prefix_encoded = self.encode(prefix, False)["input_ids"]
        targets_start: List[int] = []

        for prefix_tokens in prefix_encoded:
            targets_start.append(
                sum(
                    token != self.pad_token_id and token != self.sep_token_id
                    for token in prefix_tokens
                )
            )

        targets_end: Optional[List[int]] = None

        if suffix is None:
            sentences = [p + separator + s for p, s in zip(prefix, stimuli)]
            encoded = self.encode(sentences, manual_special=False)

        else:
            targets_end = []

            for stimuli_tokens, prefix_len in zip(
                self.encode(stimuli, False)["input_ids"], targets_start
            ):
                targets_end.append(
                    prefix_len
                    + sum(
                        token != self.cls_token_id
                        and token != self.pad_token_id
                        and token != self.sep_token_id
                        for token in stimuli_tokens
                    )
                )

            sentences = [
                pre + separator + stim + separator + suff
                for pre, stim, suff in zip(prefix, stimuli, suffix)
            ]
            encoded = self.encode(sentences, manual_special=False)

        return self.get_masked_tensors(
            encoded,
            PLL_metric=PLL_metric,
            targets_start=targets_start,
            targets_end=targets_end,
        )

    def distribution(self, batch: Iterable) -> torch.Tensor:
        """
        Returns a distribution over the vocabulary of the model.

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.

        :return: Tensor consisting of log probabilies over vocab items.
        """
        # takes in prepared text and returns scores for each sentence in batch
        token_ids, attention_masks, effective_token_ids, lengths, offsets = list(
            zip(*batch)
        )
        token_ids = torch.cat(token_ids)
        attention_masks = torch.cat(attention_masks)
        if self.device != "auto":
            token_ids = token_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
        effective_token_ids = torch.cat([torch.tensor(x) for x in effective_token_ids])

        indices = list(
            chain.from_iterable(
                [list(range(o, o + n)) for n, o in zip(lengths, offsets)]
            )
        )
        with torch.no_grad():
            output = self.model(token_ids, attention_mask=attention_masks)
            logits = output.logits[torch.arange(sum(lengths)), indices].detach()

        logprob_distribution = logits - logits.logsumexp(1).unsqueeze(1)

        return logprob_distribution

    def cloze_distribution(
        self, queries: Union[Collection[Tuple[str, str]], Tuple[str, str]]
    ) -> torch.Tensor:
        """
        Accepts as input batch of [(s_i, bw_i)] where s_i is a prompt with an
        abstract token (bw_i) representing a blank word and returns a distribution
        over the vocabulary of the model.

        :param `Iterable` queries: A batch of [(s_i, bw_i)] where s_i is a prompt with an abstract token (bw_i) representing a blank word

        :return: Tensor contisting of log probabilities over vocab items.
        """

        if len(queries) == 0:
            return torch.tensor([])

        if isinstance(next(iter(queries)), str):
            queries = [cast(Tuple[str, str], queries)]

        prompts, words = zip(*queries)

        # modified_prompts = self.add_special_tokens(prompts)
        modified_prompts = self.tokenizer.batch_decode(
            self.tokenizer(prompts)["input_ids"]
        )
        # splits = [prompt.split(word) for prompt, word in zip(modified_prompts, words)]
        splits = [
            re.split(rf"{re.escape(word)}", prompt)
            for prompt, word in zip(modified_prompts, words)
        ]
        splits = [[x.strip() for x in s] for s in splits]
        pre, post = list(zip(*splits))
        pre_idx = self.tokenizer(list(pre), add_special_tokens=False, padding=False)[
            "input_ids"
        ]
        mask_idx = [len(item) for item in pre_idx]
        # masked = [
        #     m.replace(w, self.tokenizer.mask_token)
        #     for m, w in zip(modified_prompts, words)
        # ]
        masked = [
            re.sub(rf"\b{w}\b", self.tokenizer.mask_token, m)
            for m, w in zip(prompts, words)
        ]

        with torch.no_grad():
            # encoded = self.tokenizer(
            #     masked, add_special_tokens=False, return_tensors="pt", padding=True
            # )
            encoded = self.tokenizer(masked, return_tensors="pt", padding=True)
            if self.device != "auto":
                encoded = encoded.to(self.device)
            logits = self.model(**encoded)
            presoftmax = logits.logits[torch.arange(len(queries)), mask_idx]
            if "cuda" in self.device or "auto" in self.device:
                presoftmax = presoftmax.detach().cpu()
            else:
                presoftmax = presoftmax.detach()

        logprobs = presoftmax - presoftmax.logsumexp(1).unsqueeze(1)

        return logprobs

    def entropy(
        self,
        sentence_words: Union[Collection[Tuple[str, str]]],
        space: List[List[str]] = None,
        base_two: bool = False,
    ):
        logprobs = self.cloze_distribution(sentence_words)

        if space is not None:
            probs, ranks = self.query(logprobs, space, prob=True, tolist=False)

            # stack
            equal_len = all_equal(len(x) for x in space)

            if not equal_len:
                probs = torch.nested.nested_tensor(probs).to_padded_tensor(0)
            else:
                probs = torch.stack(probs)

            # renormalize
            probs = probs / (probs.sum(1).unsqueeze(1))
            logprobs = probs.log()

        if base_two:
            H = -(logprobs.exp() * logprobs / torch.tensor(2).log()).sum(1)
        else:
            H = -(logprobs.exp() * logprobs).sum(1)

        return H

    def logprobs(
        self, batch: Iterable, rank=False
    ) -> Union[List[Tuple[torch.Tensor, str]], List[Tuple[torch.Tensor, str, int]]]:
        """
        Returns log probabilities

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.
        :param rank: Specifies whether to also return ranks of words.
        :type rank: bool

        :return: List of MLM score metrics and tokens.
        :rtype: Union[List[Tuple[torch.Tensor, str]], List[Tuple[torch.Tensor, str, int]]]
        """
        warnings.warn(
            "logprobs is deprecated, use compute_stats instead", DeprecationWarning
        )
        token_ids, attention_masks, effective_token_ids, lengths, offsets = list(
            zip(*batch)
        )
        token_ids = torch.cat(token_ids)
        attention_masks = torch.cat(attention_masks)
        if self.device != "auto":
            token_ids = token_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
        effective_token_ids = torch.cat([torch.tensor(x) for x in effective_token_ids])

        sent_tokens = list(
            map(
                lambda x: self.tokenizer.convert_ids_to_tokens(x.tolist()),
                effective_token_ids.split(lengths),
            )
        )

        indices = list(
            chain.from_iterable(
                [list(range(o, o + n)) for n, o in zip(lengths, offsets)]
            )
        )
        with torch.no_grad():
            output = self.model(token_ids, attention_mask=attention_masks)
            logits = output.logits[torch.arange(sum(lengths)), indices]
            if "cuda" in self.device:
                logits.detach().cpu()

            sent_log_probs = logits - logits.logsumexp(1).unsqueeze(1)
            if rank:
                shape = sent_log_probs.shape
                # inv_ranks = (sent_log_probs).argsort().argsort() + 1
                # ranks = shape[1] - inv_ranks + 1
                ranks = (-1.0 * sent_log_probs).argsort().argsort() + 1
                word_ranks = ranks[torch.arange(shape[0]), effective_token_ids].split(
                    lengths
                )
            sent_log_probs = (
                sent_log_probs[torch.arange(sum(lengths)), effective_token_ids]
                .type(torch.Tensor)
                .split(lengths)
            )
            # print(sent_log_probs)
            # sentence_scores = list(map(lambda x: x.sum().tolist(), logprobs))
            # outputs.append((logprobs, sent_tokens))
            if rank:
                return list(zip(sent_log_probs, sent_tokens, word_ranks))

        return list(zip(sent_log_probs, sent_tokens))

    def compute_stats(
        self,
        batch: Iterable,
        rank: bool = False,
        base_two: bool = False,
        prob: bool = False,
        return_tensors: bool = False,
    ) -> Union[
        Tuple[List[float], List[int]],
        List[float],
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Primary computational method that processes a batch of prepared sentences and returns per-token scores for each sentence. By default, returns log-probabilities.

        :param ``Iterable`` batch: batched input as processed by ``prepare_text`` or ``prime_text``.
        :param ``bool`` rank: whether the model should also return ranks per word (based on the conditional log-probability of the word in context).
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param ``bool`` return_tensors: whether the model should return scores as a list of tensors instead of a list of lists. This is important in some other convenient methods used in the package.

        :return: Either a tuple of lists, each containing probabilities and ranks per token in each sentence passed in the input.
        :rtype: ``Union[Tuple[List[float], List[float]], List[float]]``
        """
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        token_ids, attention_masks, target_token_ids, target_token_indices = zip(*batch)

        target_token_ids = torch.cat(target_token_ids)

        token_ids = torch.cat(token_ids)
        attention_masks = torch.cat(attention_masks)

        if self.device != "auto":
            token_ids = token_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)

        lengths = [len(inds) for inds in target_token_indices]

        with torch.no_grad():
            output = self.model(token_ids, attention_mask=attention_masks)
            logits = output.logits.detach()
            logits = logits[
                torch.arange(logits.shape[0]), torch.cat(target_token_indices)
            ]

        logprob_distribution = logits - logits.logsumexp(1).unsqueeze(1)

        if base_two:
            logprob_distribution = logprob_distribution / torch.tensor(2).log()

        if prob:
            logprob_distribution = logprob_distribution.exp()

        if rank:
            shape = logprob_distribution.shape
            """
            Double argsort trick:
            first argsort returns idxes of values that would return a sorted tensor,
            second argsort returns ranks (0 indexed)

            Proof: https://www.berkayantmen.com/rank.html

            TODO: Try to implement ranking in linear time but across arbitrary dimensions:
            https://stackoverflow.com/a/5284703
            """
            word_ranks = (-1.0 * logprob_distribution).argsort().argsort() + 1
            word_ranks = word_ranks[torch.arange(shape[0]), target_token_ids].split(
                lengths
            )
            word_ranks = [wr.tolist() for wr in word_ranks]

        scores = (
            logprob_distribution[
                torch.arange(logprob_distribution.size(0)), target_token_ids
            ]
            .type(torch.Tensor)
            .split(lengths)
        )
        scores = [s for s in scores]

        if not return_tensors:
            scores = [s.tolist() for s in scores]

        if rank:
            return scores, word_ranks
        else:
            return scores

    def sequence_score(
        self,
        batch,
        reduction=lambda x: x.mean(0).item(),
        prob: bool = False,
        base_two: bool = False,
        **kw,
    ):
        """
        Pooled estimates of sequence log probabilities (or some modification of it).

        :param batch: a batch of sequences whose score you want to calculate.
        :type batch: ``Union[str, List[str]]``
        :param reduction: Reduction function, is selected to be
            ``lambda x: x.mean(0).item()`` by default, which stands for the avg. log-probability per token for each sequence in the batch.
        :type reduction: Callable
        :param kw: model-specific keyword arguments to pass to the `prepare_text` function
        :return: List of floats specifying the desired score for the stimuli part of the input, e.g., P(stimuli | preamble).
        :rtype: ``List[float]``

        TODO: reduction should be a string, if it's a function, specify what kind of function. --> how to ensure it is always that type?
        """
        tokenized = self.prepare_text(batch, **kw)
        scores = self.compute_stats(
            tokenized, rank=False, base_two=base_two, prob=prob, return_tensors=True
        )
        reduced = list(map(reduction, scores))
        return reduced

    def token_score(
        self,
        batch: Union[str, List[str]],
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        rank: bool = False,
        PLL_metric: str = "original",
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        """
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)
        :param ``str`` PLL_metric: PLL scoring strategy to be used.
            Options: `original` or `within_word_l2r`. Default: `original`
            For motivation as to why to use `within_word_l2r` PLL scoring, see Kauf & Ivanova (2023):
            https://arxiv.org/abs/2305.10588

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        """
        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        tokenized = list(self.prepare_text(batch, PLL_metric=PLL_metric))
        if rank:
            scores, ranks = self.compute_stats(
                tokenized, rank=rank, prob=prob, base_two=base_two, return_tensors=True
            )
        else:
            scores = self.compute_stats(
                tokenized, prob=prob, base_two=base_two, return_tensors=True
            )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        scores = [s.tolist() for s in scores]

        tokens = [self.decode(batch[2]) for batch in tokenized]

        if rank:
            assert len(tokens) == len(scores) == len(ranks)
        else:
            assert len(tokens) == len(scores)

        res = []
        if rank:
            for t, s, r in zip(tokens, scores, ranks):
                res.append(list(zip(t, s, r)))
            # return [list(zip(t, s, r)) for t, s, r in zip(tokens, scores, ranks)]
        else:
            for t, s in zip(tokens, scores):
                res.append(list(zip(t, s)))

        return res

    def conditional_score(
        self,
        prefix: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        suffix: Union[None, str, List[str]] = None,
        separator: str = " ",
        reduction: Callable = lambda x: x.mean(0).item(),
        prob: bool = False,
        base_two: bool = False,
        **kw,
    ) -> List[float]:
        """
        Pooled estimates of sequence log probabilities (or some modification of it), given a prefix. Pooling is usually done using a function that is passed to the method.

        :param prefix: a batch of prefixes or primes passed to the
            language model. This is what the sequence is conditioned on, and the model ignores the word probabilities of this part of the input in estimating the overall score.
        :type prefix: ``Union[str, List[str]]``
        :param stimuli: a batch of sequences (same length as prefix)
            that form the main input consisting of the sequence whose
            score you want to calculate.
        :type stimuli: ``Union[str, List[str]]``
        :param reduction: Reduction function, is selected to be
            ``lambda x: x.mean(0).item()`` by default, which stands for the avg. log-probability per token for each sequence in the batch.
        :type reduction: Callable
        :param kw: model-specific keyword arguments to pass to the `prepare_text` function
        :return: List of floats specifying the desired score for the stimuli part of the input, e.g., P(stimuli | preamble).
        :rtype: ``List[float]``
        """
        primed = self.prime_text(prefix, stimuli, suffix, separator, **kw)

        result = self.compute_stats(
            primed, rank=False, base_two=base_two, prob=prob, return_tensors=True
        )
        logprob = result
        reduced = list(map(reduction, logprob))

        return reduced

    def partial_score(
        self,
        preamble: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        suffix: Union[None, str, List[str]] = None,
        separator: str = " ",
        reduction: Callable = lambda x: x.mean(0).item(),
        **kwargs,
    ) -> List[float]:
        warnings.warn(
            "partial_score is deprecated, use conditional_score instead",
            DeprecationWarning,
        )
        return self.conditional_score(
            prefix=preamble,
            stimuli=stimuli,
            suffix=suffix,
            separator=separator,
            reduction=reduction,
            **kwargs,
        )


class IncrementalLMScorer(LMScorer):
    """
    Class for Autoregressive or Incremental (or left-to-right) language models such as GPT2, etc.

    :param model: should be path to a model (.pt or .bin file) stored locally,
        or name of a pretrained model stored on the Huggingface Model Hub, or
        a model (torch.nn.Module) that have the same signature as a
        Huggingface model obtained from `AutoModelForCausalLM`. In the last
        case, a corresponding tokenizer must also be provided.
    :param device: device type that the model should be loaded on,
        options: `cpu or cuda:{0, 1, ...}`
    :type device: str, optional
    :param tokenizer: if provided, use this tokenizer.
    """

    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        device: Optional[str] = "cpu",
        tokenizer=None,
        **kwargs,
    ) -> None:
        """
        :param model: should be path to a model (.pt or .bin file) stored
            locally, or name of a pretrained model stored on the Huggingface
            Model Hub, or a model (torch.nn.Module) that have the same
            signature as a Huggingface model obtained from
            `AutoModelForCausalLM`. In the last case, a corresponding tokenizer
            must also be provided.
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        :param tokenizer: if provided, use this tokenizer.
        """
        super(IncrementalLMScorer, self).__init__(
            model, device=device, tokenizer=tokenizer, **kwargs
        )

        if isinstance(model, str):
            if self.device == "auto":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model, device_map=self.device, return_dict=True, **kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model, return_dict=True, **kwargs
                )
        else:
            self.model = model

        if self.device != "auto":
            self.model.to(self.device)

        # define CLS and SEP tokens
        if self.tokenizer.pad_token is None:
            if tokenizer is not None:
                warnings.warn(
                    "tokenizer is changed by adding pad_token_id to the tokenizer."
                )
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": ["<pad>"]}
                )
                self.tokenizer.pad_token = "<pad>"
                self.model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.padding_side == "left":
            self.tokenizer.padding_side = "right"

        if isinstance(model, str):
            self.model.eval()

        self.padding_side = self.tokenizer.padding_side

        self.padding_side = self.tokenizer.padding_side
        self.pad_token_id = self.tokenizer.pad_token_id

        # bow_subtokens, but only if the model has a
        try:
            self.bow_symbol = self.tokenizer.convert_ids_to_tokens(
                self.tokenizer(" ", add_special_tokens=False).input_ids[0]
            )[
                0
            ]  # sometimes this trick returns <bow_symbol><space>, in models like llava
        except:
            self.bow_symbol = None
        if (
            self.bow_symbol == self.tokenizer.bos_token
            or self.bow_symbol is None
            or self.bow_symbol == self.tokenizer.eos_token
        ):
            self.is_bow_tokenizer = False
        else:
            self.is_bow_tokenizer = True

        if self.is_bow_tokenizer:
            self.bow_subwords = defaultdict(lambda: False)
            for word, idx in self.tokenizer.get_vocab().items():
                if word[0] == self.bow_symbol:
                    self.bow_subwords[idx] = True
                else:
                    self.bow_subwords[idx] = False

            # for cases where the model has added tokens beyond the ones it comes with
            for idx, details in self.tokenizer.added_tokens_decoder.items():
                if details.lstrip == True:
                    self.bow_subwords[idx] = True

            self.bow_subwords = dict(self.bow_subwords)
            self.bow_subword_idx = [k for k, v in self.bow_subwords.items() if v]

    def add_special_tokens(
        self,
        text: Union[str, List[str]],
        bos_token: bool = False,
        eos_token: bool = False,
    ) -> Union[str, List[str]]:
        """
        Reformats input text to add special model-dependent tokens.

        :param text: single string or batch of strings to be
            modified.
        :type text: Union[str, List[str]]
        :param bos_token: Whether the bos_token should be added in the beginning.
        :type bos_token: bool
        :param bos_token: Whether the eos_token should be added at the end.
        :type bos_token: bool

        :return: Modified input, containing special tokens as per
            tokenizer specification
        :rtype: Union[float, List[float]]:
        """

        def _format(self, text, bos, eos):
            if bos:
                text = self.tokenizer.bos_token + text
            if eos:
                text = text + self.tokenizer.eos_token
            return text

        sentences = [text] if isinstance(text, str) else text
        sentences = [_format(text, bos_token, eos_token) for text in sentences]

        return sentences

    def encode(
        self,
        text: Union[str, List[str]],
        bos_token: bool = False,
        eos_token: bool = False,
        chat: bool = False,
    ) -> BatchEncoding:
        def _format(self, text, bos, eos):
            if bos:
                text = self.tokenizer.bos_token + text
            if eos:
                text = text + self.tokenizer.eos_token
            return text

        text = [text] if isinstance(text, str) else text
        text = [_format(self, t, bos_token, eos_token) for t in text]
        if chat == True:
            if self.tokenizer.chat_template is not None:
                encoded = self.tokenizer(
                    text, return_tensors="pt", padding=True, add_special_tokens=False
                )
            else:
                raise ValueError(
                    "Chat is set to True but the model does not have a chat template."
                )
        else:
            encoded = self.tokenizer(text, return_tensors="pt", padding=True)

        if "token_type_ids" in encoded.keys():
            encoded.pop("token_type_ids")

        return encoded

    def word_spans_tokenized(self, batch: Iterable, tokenize_function: Callable):
        """
        Aligns the spans of a string tokenized by a function to
        the tokenizer of the LM.

        :param batch: batch of sentences to be prepared for scoring.
        :param tokenize_function: the tokenizer function -- we recommend nltk.TweetTokenizer

        :return: Batch of index spans and the tokenized words.
        """
        all_spans = []
        all_lm_spans = []
        words = []
        for item in batch:
            # splitted = [tokenize_function(s) for s in item.split(" ")]
            if isinstance(tokenize_function, Callable):
                splitted = [tokenize_function(s) for s in item.split(" ")]
            else:
                splitted = tokenize_function[idx]
            words.append([" ".join(s) for s in splitted])
            spans = []
            lm_spans = []
            start = 0
            end = 0
            for i, entry in enumerate(splitted):
                end = len(entry)
                old_end = start + end
                spans.append((start, old_end))
                start += end

                joined = ' '.join(entry)
                if i != 0:
                    joined = f' {joined}'

                # mapping from tokenizer spans to LM tokenizer ids
                lm_tokenized = self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer(joined, add_special_tokens=False)[
                        "input_ids"
                    ]
                )
                len_diff = len(lm_tokenized) - len(entry)
                if i == 0:
                    new_start = 0
                else:
                    new_start = lm_spans[i - 1][-1]

                if len_diff > 0:
                    new_end = new_start + 1 + len_diff
                else:
                    new_end = new_start + 1
                lm_spans.append((new_start, new_end))
            all_spans.append(spans)
            all_lm_spans.append(lm_spans)
            # words.append([s[0] for s in splitted])

        return all_lm_spans, words

    def prepare_text(
        self,
        text: Union[str, List[str], BatchEncoding],
        bos_token: bool = False,
        eos_token: bool = False,
        chat: bool = False,
    ) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on.

        :param text: batch of sentences to be prepared for scoring.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        if isinstance(text, BatchEncoding):
            encoded = text
        else:
            encoded = self.encode(text, bos_token, eos_token, chat)
        offsets = [0] * len(encoded["input_ids"])
        return encoded, offsets

    def prime_text(
        self,
        preamble: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        separator: Union[str, List[str]] = " ",
        bos_token: bool = False,
        eos_token: bool = False,
        chat: bool = False,
    ) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on.

        :param ``Union[str, List[str]]`` preamble: Batch of prefixes/prime/preambles on which the LM is conditioned.
        :param ``Union[str, List[str]]`` stimuli: Batch of continuations that are scored based on the conditioned text (provided in the ``preamble``). The positions of the elements match their counterparts in the ``preamble``.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        preamble_text = [preamble] if isinstance(preamble, str) else preamble
        if chat == True:
            if self.tokenizer.chat_template is not None:
                preamble_encoded = self.tokenizer(
                    preamble_text, add_special_tokens=False
                )["input_ids"]
            else:
                raise ValueError(
                    "Chat is set to True but the model does not have a chat template."
                )
        else:
            preamble_encoded = self.tokenizer(preamble_text)["input_ids"]
        preamble_lens = []
        for preamble_tokens in preamble_encoded:
            if bos_token:
                restricted_id = float("inf")
                bos_offset = 1
            else:
                restricted_id = self.tokenizer.pad_token_id
                bos_offset = 0
            preamble_lens.append(
                len([token for token in preamble_tokens if token != restricted_id])
                - 1
                + bos_offset
            )

        if isinstance(separator, str):
            sentences = (
                [preamble + separator + stimuli]
                if isinstance(preamble, str)
                else [p + separator + s for p, s in list(zip(preamble, stimuli))]
            )

        elif isinstance(separator, list):
            assert not isinstance(preamble, str)
            assert len(preamble) == len(separator) == len(stimuli)

            sentences = [
                p + sep + s for p, sep, s in list(zip(preamble, separator, stimuli))
            ]

        return self.encode(sentences, bos_token, eos_token, chat), preamble_lens

    def distribution(self, batch: Iterable) -> torch.Tensor:
        """
        Returns a distribution over the vocabulary of the model.

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.

        :return: Tensor consisting of log probabilies over vocab items.
        """
        batch, offsets = batch
        ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        if self.device != "auto":
            ids = ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_masks)
            logits = outputs.logits
            if "cuda" in self.device or "auto" in self.device:
                logits.detach()

        outputs = []
        for sent_index in range(len(ids)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(batch.tokens(sent_index))
                if sent_nopad_mask[i] and i > offsets[sent_index] + 1
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            # ignore first token (<|eos|>)
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            # sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")

            outputs.append(sent_logits[-1])
        return torch.stack(outputs, 0)

    def next_word_distribution(
        self, queries: List, bos_token=False, eos_token=False, surprisal: bool = False, chat: bool = False
    ):
        """
        Returns the log probability distribution of the next word.
        """
        encoded = self.encode(queries, bos_token, eos_token, chat=chat)
        if self.device != "auto":
            encoded = encoded.to(self.device)
        query_ids = [
            [j for j, i in enumerate(instance) if i != self.tokenizer.pad_token_id][-1]
            for instance in encoded["input_ids"].tolist()
        ]

        with torch.no_grad():
            logits = self.model(**encoded).logits.detach()
        # logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits[torch.arange(len(query_ids)), query_ids]
        logprobs = logits - logits.logsumexp(1).unsqueeze(1)

        if surprisal:
            logprobs = -1.0 * logprobs

        return logprobs

    def compute_stats(
        self,
        batch: Iterable,
        rank: bool = False,
        prob: bool = False,
        base_two: bool = False,
        return_tensors: bool = False,
        bow_correction: bool = False,
    ) -> Union[Tuple[List[float], List[float]], List[float]]:
        """
        Primary computational method that processes a batch of prepared sentences and returns per-token scores for each sentence. By default, returns log-probabilities.

        :param ``Iterable`` batch: batched input as processed by ``prepare_text`` or ``prime_text``.
        :param ``bool`` rank: whether the model should also return ranks per word (based on the conditional log-probability of the word in context).
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param ``bool`` return_tensors: whether the model should return scores as a list of tensors instead of a list of lists. This is important in some other convenient methods used in the package.
        :param ``bool'' bow_correction: whether to apply the beginning of word correction, as pointed out in Pimentel and Meister (2024) and Oh and Schuler (2024).

        :return: Either a tuple of lists, each containing probabilities and ranks per token in each sentence passed in the input.
        :rtype: ``Union[Tuple[List[float], List[int]], List[float]]``
        """
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        encoded, offsets = batch
        if self.device != "auto":
            encoded = encoded.to(self.device)

        # ids = [
        #     [i for i in instance if i != self.tokenizer.pad_token_id]
        #     for instance in encoded["input_ids"].tolist()
        # ]
        ids = [
            [i for i, am in zip(instance, attention_mask) if am != 0]
            for instance, attention_mask in zip(
                encoded["input_ids"].tolist(), encoded["attention_mask"].tolist()
            )
        ]

        ## Ignore the probabilities of the first token.
        effective_ids = [id[1:] for id in ids]

        with torch.no_grad():
            logits = self.model(**encoded).logits.detach()

        # logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits.split([1] * len(offsets))

        ## Set up storage variables
        scores = []
        if rank:
            ranks = []

        for logit, idx, offset in zip(logits, effective_ids, offsets):
            length = len(idx)
            # logit = logit.squeeze(0)[torch.arange(offset, length),]

            # logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)

            query_ids = idx[offset:]
            logit = logit.squeeze(0)
            logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)

            actual_logprob_distribution = logprob_distribution[
                torch.arange(offset, length),
            ]

            score = actual_logprob_distribution[
                torch.arange(length - offset), query_ids
            ]

            if not self.is_bow_tokenizer:
                bow_correction = False

            if bow_correction:
                # mask_forward = torch.zeros_like(logprob_distribution)
                # mask_current = torch.zeros_like(logprob_distribution)
                mask_forward = torch.zeros(length).to(self.device)
                mask_current = torch.zeros(length).to(self.device)

                for i in range(len(idx)):
                    if i == len(idx) - 1:
                        mask_forward[i] = 1
                        if not self.bow_subwords[idx[i]]:
                            mask_current[i] = 0
                        else:
                            mask_current[i] = 1
                        break
                    elif self.bow_subwords[idx[i + 1]]:
                        mask_forward[i] = 1
                        if not self.bow_subwords[idx[i]]:
                            mask_current[i] = 0
                        else:
                            mask_current[i] = 1
                    else:
                        mask_forward[i] = 0
                        mask_current[i] = 1

                mask_forward = mask_forward[offset:]
                # mask_current = mask_current[offset:]
                mask_current = torch.roll(mask_forward, shifts=1)
                mask_current[0] = 0.0

                bow_subword_idx_tensor = torch.tensor(self.bow_subword_idx).to(
                    self.device
                )

                forward_correction = (
                    logprob_distribution[offset:][torch.arange(length - offset) + 1,]
                    .index_select(-1, bow_subword_idx_tensor)
                    .logsumexp(1)
                )

                current_correction = (
                    actual_logprob_distribution[torch.arange(length - offset),]
                    .index_select(-1, bow_subword_idx_tensor)
                    .logsumexp(1)
                )

                score = (
                    score
                    + (forward_correction * mask_forward)
                    - (current_correction * mask_current)
                )

            if base_two:
                """
                Log_2(X) = log_e(X)/log_e(2) (broadcasted)
                """
                score = score / torch.tensor(2).log()
            else:
                if prob:
                    score = score.exp()
                else:
                    score = score

            if rank:
                # shape = logprob_distribution.shape
                """
                Double argsort trick:
                first argsort returns idxes of values that would return a sorted tensor,
                second argsort returns ranks (0 indexed)

                Proof: https://www.berkayantmen.com/rank.html

                TODO: Try to implement ranking in linear time but across arbitrary dimensions:
                https://stackoverflow.com/a/5284703
                """
                word_ranks = (
                    -1.0 * actual_logprob_distribution
                ).argsort().argsort() + 1
                # inv_ranks = logprob_distribution.argsort().argsort() + 1
                # word_ranks = shape[1] - inv_ranks + 1
                word_ranks = word_ranks[
                    torch.arange(length - offset), query_ids
                ].tolist()
                ranks.append(word_ranks)

            scores.append(score)

        if not return_tensors:
            # scores = [torch.tensor(l).detach() for l in scores]
            scores = [s.tolist() for s in scores]

        if rank:
            return scores, ranks
        else:
            return scores

    def sequence_score(
        self,
        batch,
        reduction=lambda x: x.mean(0).item(),
        base_two=False,
        bow_correction=False,
        **kwargs,
    ):
        """
        TODO: reduction should be a string, if it's a function, specify what kind of function. --> how to ensure it is always that type?
        """
        tokenized = self.prepare_text(batch, **kwargs)
        scores = self.compute_stats(
            tokenized,
            rank=False,
            base_two=base_two,
            bow_correction=bow_correction,
            return_tensors=True,
        )
        reduced = list(map(reduction, scores))
        return reduced

    def token_score(
        self,
        batch: Union[str, List[str]],
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        rank: bool = False,
        decode: bool = False,
        bow_correction: bool = False,
        **kwargs,
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        """
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)
        :param ``bool'' bow_correction: whether to apply the beginning of word correction, as pointed out in Pimentel and Meister (2024) and Oh and Schuler (2024).

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        """

        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        tokenized = self.prepare_text(batch, **kwargs)
        if rank:
            scores, ranks = self.compute_stats(
                tokenized,
                rank=rank,
                prob=prob,
                base_two=base_two,
                bow_correction=bow_correction,
                return_tensors=True,
            )
        else:
            scores = self.compute_stats(
                tokenized,
                prob=prob,
                base_two=base_two,
                bow_correction=bow_correction,
                return_tensors=True,
            )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        scores = [s.tolist() for s in scores]

        # indices = [
        #     [i for i in indexed if i != self.tokenizer.pad_token_id]
        #     for indexed in tokenized[0]["input_ids"].tolist()
        # ]

        indices = [
            [i for i, am in zip(instance, attention_mask) if am != 0]
            for instance, attention_mask in zip(
                tokenized[0]["input_ids"].tolist(),
                tokenized[0]["attention_mask"].tolist(),
            )
        ]
        if decode:
            tokens = [self.decode(idx) for idx in indices]
        else:
            tokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in indices]

        if rank:
            assert len(tokens) == len(scores) == len(ranks)
        else:
            assert len(tokens) == len(scores)

        res = []
        if rank:
            for t, s, r in zip(tokens, scores, ranks):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    ra = [0] * diff + r
                    res.append(list(zip(t, sc, ra)))
                else:
                    res.append(list(zip(t, sc, ra)))
            # return [list(zip(t, s, r)) for t, s, r in zip(tokens, scores, ranks)]
        else:
            for t, s in zip(tokens, scores):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    res.append(list(zip(t, sc)))
                else:
                    res.append(list(zip(t, sc)))

        return res

    def word_score_tokenized(
        self,
        batch: Iterable,
        tokenize_function: Callable,
        bow_correction: bool = False,
        bos_token: bool = False,
        eos_token: bool = False,
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        return_tensors: bool = False,
    ):
        """
        Returns the logprobs per word, as tokenized by the tokenize_function.

        :param batch: batch of sentences to be prepared for scoring.
        :param tokenize_function: tokenize function to maps strings to tokenized words.
        :param bow_correction: whether to apply Oh and Schuler's correction.
        :param bos_token: if a beginning of sentence token should be added (specify this as True for GPT2 and Pythia, for other they do it by default).
        :param eos_token: if an end of sentence token should be added
        :param prob: if the scores should be probabilities instead of logprobabilities
        :param base_two: if the logprobabilities should

        :return: Batch of words and their corresponding log-probs (summed log-probs of the tokens)
        """
        batch = [batch] if isinstance(batch, str) else batch

        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        tokenized = [" ".join(tokenize_function(item)) for item in batch]

        encoded, offsets = self.prepare_text(
            batch, bos_token=bos_token, eos_token=eos_token
        )

        scores = self.compute_stats(
            (encoded, offsets),
            bow_correction=bow_correction,
            prob=prob,
            base_two=base_two,
            return_tensors=True,
        )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        spans, words = self.word_spans_tokenized(tokenized, tokenize_function)

        word_scores = []
        for i, span in enumerate(spans):
            ws = []
            for j, (s, e) in enumerate(span):
                if prob:
                    score = scores[i][s:e].prod()
                else:
                    score = scores[i][s:e].sum()
                if not return_tensors:
                    score = score.item()

                ws.append((words[i][j], score))
            word_scores.append(ws)

        return word_scores

    def conditional_word_score_tokenized(
        self,
        prefix,
        stimuli,
        tokenize_function: Callable,
        separator=" ",
        bow_correction=False,
        bos_token=False,
        eos_token=False,
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        return_tensors: bool = False,
    ):

        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        prefix = [prefix] if isinstance(prefix, str) else prefix
        stimuli = [stimuli] if isinstance(stimuli, str) else stimuli

        if isinstance(tokenize_function, Callable):
            tokenized = [" ".join(tokenize_function(item)) for item in stimuli]
        elif isinstance(tokenize_function, list):
            tokenize_function = [[[tt] for tt in t] for t in tokenize_function]
            tokenized = [" ".join([tt[0] for tt in t]) for t in tokenize_function]
        else:
            raise ValueError("Incorrect type of tokenize_function argument (either callable or list)")

        encoded, offsets = self.prime_text(
            prefix,
            stimuli,
            separator=separator,
            bos_token=bos_token,
            eos_token=eos_token,
        )

        scores = self.compute_stats(
            (encoded, offsets),
            bow_correction=bow_correction,
            prob=prob,
            base_two=base_two,
            return_tensors=True,
        )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        spans, words = self.word_spans_tokenized(tokenized, tokenize_function)

        word_scores = []
        for i, span in enumerate(spans):
            ws = []
            for j, (s, e) in enumerate(span):
                if prob:
                    score = scores[i][s:e].prod()
                else:
                    score = scores[i][s:e].sum()
                if not return_tensors:
                    score = score.item()

                ws.append((words[i][j], score))
            word_scores.append(ws)

        return word_scores

    def logprobs(self, batch: Iterable, rank=False) -> Union[float, List[float]]:
        """
        Returns log probabilities

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.
        :param rank: Specifies whether to also return ranks of words.
        :type rank: bool

        :return: List of LM score metrics (probability and rank)
            and tokens.
        :rtype: Union[List[Tuple[torch.Tensor, str]], List[Tuple[torch.Tensor, str, int]]]
        """
        warnings.warn(
            "logprobs is deprecated, use compute_stats instead", DeprecationWarning
        )
        batch, offsets = batch
        ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        if self.device != "auto":
            ids = ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_masks)
            logits = outputs.logits
            if "cuda" in self.device or "auto" in self.device:
                logits.detach()

        outputs = []
        for sent_index in range(len(ids)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(batch.tokens(sent_index))
                if sent_nopad_mask[i] and i > offsets[sent_index]
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            # ignore first token (<|eos|>)
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")
            # ids_scores.shape = [seq_len + 1]
            # select only the ids present in the sentence out of all vocab items (as a 2d array)
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            # log_prob.shape = [seq_len + 1]
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)

            sent_log_probs = sent_log_probs.type(torch.Tensor)
            sent_log_probs = sent_log_probs[offsets[sent_index] :]
            lengths = len(sent_log_probs)
            if rank:
                shape = sent_logits.shape
                inv_ranks = (sent_logits).argsort().argsort() + 1
                ranks = shape[1] - inv_ranks + 1
                word_ranks = ranks[
                    list(range(shape[0]))[offsets[sent_index] :],
                    sent_ids[offsets[sent_index] :].tolist(),
                ].split(lengths)
                word_ranks = [x[0] for x in word_ranks]
                outputs.append((sent_log_probs, sent_tokens, word_ranks))
            else:
                outputs.append((sent_log_probs, sent_tokens))
            # output = (sent_log_probs.sum(), sent_ids, sent_tokens)
            # outputs.append(output)
        return outputs

    def fixed_label_score(
        self,
        batch: Iterable,
        labels: Iterable,
        reduction=lambda x: x.sum(0),
        inference=False,
        probs=False,
    ) -> Union[Tuple[str, float], List[float]]:
        """
        Returns log probabilities for a fixed set of labels (continuation)
        given a batch of prefixes.

        :param `Iterable` batch: Batch of inputs.
        :param `Iterable` labels: Label strings.
        :param reduction: reduction function.
        :param inference: whether or not to return argmax labels.
        :param probs: whether or nor to return relative probabilities.

        :return: List of LM score metrics (probability and rank)
            and tokens.
        :rtype: Union[Tuple[str, float], List[float]]
        """
        labels_with_space = [f" {l.strip()}" for l in labels]
        label_ids = self.tokenizer(labels_with_space, padding=True).input_ids
        label_ids_unzipped = list(zip(*label_ids))

        filter_ids = [
            list(range(len([i for i in j if i != self.tokenizer.pad_token_id])))
            for j in label_ids
        ]

        self.tokenizer.padding_side = "left"
        tokenized = self.tokenizer(batch, return_tensors="pt", padding=True)
        if self.device != "auto":
            tokenized = tokenized.to(self.device)

        outputs = self.model.generate(
            **tokenized,
            max_new_tokens=len(label_ids_unzipped),
            return_dict_in_generate=True,
            output_scores=True,
        )

        logprobs = []
        for i, score in enumerate(outputs.scores):
            score = score - score.logsumexp(1).unsqueeze(1)
            logprobs.append(score[:, label_ids_unzipped[i]])

        logprobs = batch_wise_logprobs(logprobs, filter_ids, reduction)

        self.tokenizer.padding_side = self.padding_side

        if probs:
            logprobs = logprobs.softmax(1)

        if inference:
            predictions = logprobs.argmax(1)
            predicted_labels = [labels[i] for i in predictions.tolist()]
            lps = logprobs[torch.arange(len(batch)), predictions].tolist()
            return predicted_labels, lps
        else:
            return logprobs


class Seq2SeqScorer(LMScorer):
    """
    Class for Autoregressive or Incremental (or left-to-right) language models such as GPT2, etc.

    :param model: should be path to a model (.pt or .bin file) stored locally,
        or name of a pretrained model stored on the Huggingface Model Hub, or
        a model (torch.nn.Module) that have the same signature as a
        Huggingface model obtained from `AutoModelForSeq2SeqLM`. In the last
        case, a corresponding tokenizer must also be provided.
    :param device: device type that the model should be loaded on,
        options: `cpu or cuda:{0, 1, ...}`
    :type device: str, optional
    :param tokenizer: if provided, use this tokenizer.
    """

    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        device: Optional[str] = "cpu",
        tokenizer=None,
        **kwargs,
    ) -> None:
        """
        :param model: should be path to a model (.pt or .bin file) stored
            locally, or name of a pretrained model stored on the Huggingface
            Model Hub, or a model (torch.nn.Module) that have the same
            signature as a Huggingface model obtained from
            `AutoModelForSeq2SeqLM`. In the last case, a corresponding
            tokenizer must also be provided.
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        :param tokenizer: if provided, use this tokenizer.
        """
        super(Seq2SeqScorer, self).__init__(model, device=device, tokenizer=tokenizer)

        if isinstance(model, str):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model, return_dict=True, **kwargs
            )
        else:
            self.model = model

        # define CLS and SEP tokens
        if self.tokenizer.pad_token is None:
            if tokenizer is not None:
                warnings.warn(
                    "tokenizer is changed by adding pad_token to the tokenizer."
                )
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|pad|>"]}
            )
            self.tokenizer.pad_token = "<|pad|>"

        if self.tokenizer.bos_token is None:
            if tokenizer is not None:
                warnings.warn(
                    "tokenizer is changed by adding bos_token to the tokenizer."
                )
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|bos|>"]}
            )
            self.tokenizer.bos_token = "<|bos|>"

        if isinstance(model, str):
            self.model.resize_token_embeddings(len(self.tokenizer))
            if self.device != "auto":
                self.model.to(self.device)
            self.model.eval()

        self.decoder_start_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.decoder_start_token_id
        )

    def add_special_tokens(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Reformats input text to add special model-dependent tokens.

        :param text: single string or batch of strings to be
            modified.
        :type text: Union[str, List[str]]

        :return: Modified input, containing special tokens as per
            tokenizer specification
        :rtype: Union[float, List[float]]:
        """
        sentences = [text] if isinstance(text, str) else text
        sentences = [self.tokenizer.bos_token + sentence for sentence in sentences]

        return sentences

    def encode(self, text: Union[str, List[str]]) -> BatchEncoding:
        text = [text] if isinstance(text, str) else text
        return self.tokenizer(text, return_tensors="pt", padding=True)

    def prepare_text(self, text: Union[str, List[str], BatchEncoding]) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on.

        :param text: batch of sentences to be prepared for scoring.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        if isinstance(text, BatchEncoding):
            encoded = text
        else:
            encoded = self.encode(text)
        offsets = [0] * len(encoded["input_ids"])
        return encoded, offsets

    def prime_text(self, prefix: Union[str, List[str]], stimuli: Union[str, List[str]]):

        prefix = [prefix] if isinstance(prefix, str) else prefix
        stimuli = [stimuli] if isinstance(stimuli, str) else stimuli

        stimuli = [f"{self.decoder_start_token} {s}" for s in stimuli]

        prefix_encoded = self.tokenizer(
            prefix, return_tensors="pt", padding=True, add_special_tokens=False
        )
        stimuli_encoded = self.tokenizer(
            stimuli, return_tensors="pt", padding=True, add_special_tokens=False
        )
        offset = [0] * len(prefix_encoded["input_ids"])  # this will be ignored anyway

        return prefix_encoded, stimuli_encoded, offset

    def compute_conditional_stats(
        self,
        batch: Iterable,
        rank: bool = False,
        prob: bool = False,
        base_two: bool = False,
        return_tensors: bool = False,
    ):
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        prefix, stimuli, offsets = batch

        if self.device != "auto":
            prefix = prefix.to(self.device)
            stimuli = stimuli.to(self.device)

        ids = [
            [i for i, am in zip(instance, attention_mask) if am != 0]
            for instance, attention_mask in zip(
                stimuli["input_ids"].tolist(), stimuli["attention_mask"].tolist()
            )
        ]

        # Ignore the probabilities of the first token.
        effective_ids = [id[1:] for id in ids]

        with torch.no_grad():
            logits = self.model(
                **prefix, decoder_input_ids=stimuli["input_ids"]
            ).logits.detach()

        scores = []
        if rank:
            ranks = []

        for logit, idx, offset in zip(logits, effective_ids, offsets):
            length = len(idx)
            logit = logit.squeeze(0)[torch.arange(offset, length),]

            logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)
            query_ids = idx[offset:]
            if base_two:
                """
                Log_2(X) = log_e(X)/log_e(2) (broadcasted)
                """
                score = (
                    logprob_distribution[torch.arange(length - offset), query_ids]
                    / torch.tensor(2).log()
                ).tolist()
            else:
                if prob:
                    score = (
                        logprob_distribution[torch.arange(length - offset), query_ids]
                        .exp()
                        .tolist()
                    )
                else:
                    score = logprob_distribution[
                        torch.arange(length - offset), query_ids
                    ].tolist()

            if rank:
                # shape = logprob_distribution.shape
                """
                Double argsort trick:
                first argsort returns idxes of values that would return a sorted tensor,
                second argsort returns ranks (0 indexed)

                Proof: https://www.berkayantmen.com/rank.html

                TODO: Try to implement ranking in linear time but across arbitrary dimensions:
                https://stackoverflow.com/a/5284703
                """
                word_ranks = (-1.0 * logprob_distribution).argsort().argsort() + 1
                # inv_ranks = logprob_distribution.argsort().argsort() + 1
                # word_ranks = shape[1] - inv_ranks + 1
                word_ranks = word_ranks[
                    torch.arange(length - offset), query_ids
                ].tolist()
                ranks.append(word_ranks)

            scores.append(score)

        if return_tensors:
            scores = [torch.tensor(l) for l in scores]

        if rank:
            return scores, ranks
        else:
            return scores

    def conditional_score(
        self,
        prefix: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        reduction: Callable = lambda x: x.mean(0).item(),
        prob: bool = False,
        base_two: bool = False,
    ):
        primed = self.prime_text(prefix, stimuli)

        result = self.compute_conditional_stats(
            primed, rank=False, base_two=base_two, prob=prob, return_tensors=True
        )
        logprob = result
        reduced = list(map(reduction, logprob))

        return reduced

    def conditional_token_score(
        self,
        prefix: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        rank: bool = False,
        decode: bool = True,
    ):
        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        # tokenized = self.prepare_text(batch, **kwargs)
        tokenized = self.prime_text(prefix, stimuli)
        if rank:
            scores, ranks = self.compute_conditional_stats(
                tokenized, rank=rank, prob=prob, base_two=base_two, return_tensors=True
            )
        else:
            scores = self.compute_conditional_stats(
                tokenized, prob=prob, base_two=base_two, return_tensors=True
            )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        scores = [s.tolist() for s in scores]

        # indices = [
        #     [i for i in indexed if i != self.tokenizer.pad_token_id]
        #     for indexed in tokenized[0]["input_ids"].tolist()
        # ]

        indices = [
            [i for i, am in zip(instance, attention_mask) if am != 0]
            for instance, attention_mask in zip(
                tokenized[1]["input_ids"].tolist(),
                tokenized[1]["attention_mask"].tolist(),
            )
        ]
        if decode:
            tokens = [self.decode(idx) for idx in indices]
        else:
            tokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in indices]

        if rank:
            assert len(tokens) == len(scores) == len(ranks)
        else:
            assert len(tokens) == len(scores)

        res = []
        if rank:
            for t, s, r in zip(tokens, scores, ranks):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    ra = [0] * diff + r
                    res.append(list(zip(t, sc, ra)))
                else:
                    res.append(list(zip(t, sc, ra)))
            # return [list(zip(t, s, r)) for t, s, r in zip(tokens, scores, ranks)]
        else:
            for t, s in zip(tokens, scores):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    res.append(list(zip(t, sc)))
                else:
                    res.append(list(zip(t, sc)))

        return res

    def prime_text_deprecated(
        self,
        preamble: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        separator=" ",
    ) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on.

        :param ``Union[str, List[str]]`` preamble: Batch of prefixes/prime/preambles on which the LM is conditioned.
        :param ``Union[str, List[str]]`` stimuli: Batch of continuations that are scored based on the conditioned text (provided in the ``preamble``). The positions of the elements match their counterparts in the ``preamble``.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        preamble_text = [preamble] if isinstance(preamble, str) else preamble
        preamble_encoded = self.tokenizer(preamble_text)["input_ids"]
        preamble_lens = []
        for preamble_tokens in preamble_encoded:
            preamble_lens.append(
                len(
                    [
                        token
                        for token in preamble_tokens
                        if token != self.tokenizer.pad_token_id
                        and token != self.tokenizer.sep_token_id
                    ]
                )
                - 1
            )

        sentences = (
            [preamble + " " + stimuli]
            if isinstance(preamble, str)
            else [p + " " + s for p, s in list(zip(preamble, stimuli))]
        )

        return self.encode(sentences), preamble_lens

    def distribution(self, batch: Iterable) -> torch.Tensor:
        """
        Returns a distribution over the vocabulary of the model.

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.

        :return: Tensor consisting of log probabilies over vocab items.
        """
        batch, offsets = batch
        ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        if self.device != "auto":
            ids = ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_masks)
            logits = outputs.logits
            if "cuda" in self.device:
                logits.detach()

        outputs = []
        for sent_index in range(len(ids)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(batch.tokens(sent_index))
                if sent_nopad_mask[i] and i > offsets[sent_index] + 1
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            # ignore first token (<|eos|>)
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")

            outputs.append(sent_logits[-1])
        return torch.stack(outputs, 0)

    def next_word_distribution(self, queries: List, surprisal: bool = False):
        """
        Returns the log probability distribution of the next word.
        """
        encoded = self.encode(queries)
        if self.device != "auto":
            encoded = encoded.to(self.device)
        query_ids = [
            [j for j, i in enumerate(instance) if i != self.tokenizer.pad_token_id][-1]
            for instance in encoded["input_ids"].tolist()
        ]

        logits = self.model(**encoded).logits.detach()
        # logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits[torch.arange(len(query_ids)), query_ids]
        logprobs = logits - logits.logsumexp(1).unsqueeze(1)

        if surprisal:
            logprobs = -1.0 * logprobs

        return logprobs

    def compute_stats(
        self,
        batch: Iterable,
        source: Iterable,
        rank: bool = False,
        prob: bool = False,
        base_two: bool = False,
        return_tensors: bool = False,
    ) -> Union[Tuple[List[float], List[float]], List[float]]:
        """
        Primary computational method that processes a batch of prepared sentences and returns per-token scores for each sentence. By default, returns log-probabilities.

        :param ``Iterable`` batch: batched input as processed by ``prepare_text`` or ``prime_text``.
        :param ``bool`` rank: whether the model should also return ranks per word (based on the conditional log-probability of the word in context).
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param ``bool`` return_tensors: whether the model should return scores as a list of tensors instead of a list of lists. This is important in some other convenient methods used in the package.

        :return: Either a tuple of lists, each containing probabilities and ranks per token in each sentence passed in the input.
        :rtype: ``Union[Tuple[List[float], List[int]], List[float]]``
        """
        raise Exception(
            "This function is currently erroneous and a fix is in progress. Apologies for the inconvenience."
        )
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        source_encoded, source_offsets = source
        target_encoded, target_offsets = batch
        if self.device != "auto":
            source_ids = source_encoded["input_ids"].to(self.device)
            target_ids = target_encoded["input_ids"].to(self.device)

        source_ids_list = [
            [i for i in instance if i != self.tokenizer.pad_token_id]
            for instance in source_encoded["input_ids"].tolist()
        ]
        target_ids_list = [
            [i for i in instance if i != self.tokenizer.pad_token_id]
            for instance in target_encoded["input_ids"].tolist()
        ]

        ## Ignore the probabilities of the first token.
        source_effective_ids = [id[1:] for id in source_ids_list]
        target_effective_ids = [id[1:] for id in target_ids_list]

        with torch.no_grad():
            logits = self.model(input_ids=source_ids, labels=target_ids).logits.detach()

        # logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits.split([1] * len(target_offsets))

        ## Set up storage variables
        scores = []
        if rank:
            ranks = []

        for logit, idx, offset in zip(logits, target_effective_ids, target_offsets):
            length = len(idx)
            logit = logit.squeeze(0)[:, :-1][torch.arange(offset, length),]

            logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)
            query_ids = idx[offset:]
            if base_two:
                """
                Log_2(X) = log_e(X)/log_e(2) (broadcasted)
                """
                score = (
                    logprob_distribution[torch.arange(length - offset), query_ids]
                    / torch.tensor(2).log()
                ).tolist()
            else:
                if prob:
                    score = (
                        logprob_distribution[torch.arange(length - offset), query_ids]
                        .exp()
                        .tolist()
                    )
                else:
                    score = logprob_distribution[
                        torch.arange(length - offset), query_ids
                    ].tolist()

            if rank:
                # shape = logprob_distribution.shape
                """
                Double argsort trick:
                first argsort returns idxes of values that would return a sorted tensor,
                second argsort returns ranks (0 indexed)

                Proof: https://www.berkayantmen.com/rank.html

                TODO: Try to implement ranking in linear time but across arbitrary dimensions:
                https://stackoverflow.com/a/5284703
                """
                word_ranks = (-1.0 * logprob_distribution).argsort().argsort() + 1
                # inv_ranks = logprob_distribution.argsort().argsort() + 1
                # word_ranks = shape[1] - inv_ranks + 1
                word_ranks = word_ranks[
                    torch.arange(length - offset), query_ids
                ].tolist()
                ranks.append(word_ranks)

            scores.append(score)

        if return_tensors:
            scores = [torch.tensor(l) for l in scores]

        if rank:
            return scores, ranks
        else:
            return scores

    def sequence_score(
        self,
        batch,
        reduction=lambda x: x.mean(0).item(),
        base_two=False,
        source_format="blank",
        source=None,
    ):
        """
        TODO: reduction should be a string, if it's a function, specify what kind of function. --> how to ensure it is always that type?
        """
        raise Exception(
            "This function is currently erroneous and a fix is in progress. Apologies for the inconvenience."
        )
        if source is not None:
            assert len(source) == len(batch)
            source_format = "custom"

        tokenized = self.prepare_text(batch)
        if source_format == "blank":
            source = [""] * len(batch)
        elif source_format == "copy":
            source = batch
        source = self.prepare_text(source)

        scores = self.compute_stats(
            tokenized, source, rank=False, base_two=base_two, return_tensors=True
        )
        reduced = list(map(reduction, scores))
        return reduced

    def token_score(
        self,
        batch: Union[str, List[str]],
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        rank: bool = False,
        source_format: str = "blank",
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        """
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        """
        raise Exception(
            "This function is currently erroneous and a fix is in progress. Apologies for the inconvenience."
        )
        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        tokenized = self.prepare_text(batch)
        if source_format == "blank":
            source = [""] * len(batch)
        elif source_format == "copy":
            source = batch
        source = self.prepare_text(source)

        if rank:
            scores, ranks = self.compute_stats(
                tokenized,
                source,
                rank=rank,
                prob=prob,
                base_two=base_two,
                return_tensors=True,
            )
        else:
            scores = self.compute_stats(
                tokenized, source, prob=prob, base_two=base_two, return_tensors=True
            )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        scores = [s.tolist() for s in scores]

        indices = [
            [i for i in indexed if i != self.tokenizer.pad_token_id]
            for indexed in tokenized[0]["input_ids"].tolist()
        ]
        tokens = [self.decode(idx) for idx in indices]

        if rank:
            assert len(tokens) == len(scores) == len(ranks)
        else:
            assert len(tokens) == len(scores)

        res = []
        if rank:
            for t, s, r in zip(tokens, scores, ranks):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    ra = [0] * diff + r
                    res.append(list(zip(t, sc, ra)))
                else:
                    res.append(list(zip(t, sc, ra)))
            # return [list(zip(t, s, r)) for t, s, r in zip(tokens, scores, ranks)]
        else:
            for t, s in zip(tokens, scores):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    res.append(list(zip(t, sc)))
                else:
                    res.append(list(zip(t, sc)))

        return res

    def logprobs(
        self, batch: Iterable, rank=False, source_format: str = "blank"
    ) -> Union[float, List[float]]:
        """
        Returns log probabilities

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.
        :param rank: Specifies whether to also return ranks of words.
        :type rank: bool

        :return: List of LM score metrics (probability and rank)
            and tokens.
        :rtype: Union[List[Tuple[torch.Tensor, str]], List[Tuple[torch.Tensor, str, int]]]
        """
        warnings.warn(
            "logprobs is deprecated, use compute_stats instead", DeprecationWarning
        )
        batch, offsets = batch
        ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        if self.device != "auto":
            ids = ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_masks)
            logits = outputs.logits
            if "cuda" in self.device:
                logits.detach()

        outputs = []
        for sent_index in range(len(ids)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(batch.tokens(sent_index))
                if sent_nopad_mask[i] and i > offsets[sent_index]
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            # ignore first token (<|eos|>)
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")
            # ids_scores.shape = [seq_len + 1]
            # select only the ids present in the sentence out of all vocab items (as a 2d array)
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            # log_prob.shape = [seq_len + 1]
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)

            sent_log_probs = sent_log_probs.type(torch.Tensor)
            sent_log_probs = sent_log_probs[offsets[sent_index] :]
            lengths = len(sent_log_probs)
            if rank:
                shape = sent_logits.shape
                inv_ranks = (sent_logits).argsort().argsort() + 1
                ranks = shape[1] - inv_ranks + 1
                word_ranks = ranks[
                    list(range(shape[0]))[offsets[sent_index] :],
                    sent_ids[offsets[sent_index] :].tolist(),
                ].split(lengths)
                word_ranks = [x[0] for x in word_ranks]
                outputs.append((sent_log_probs, sent_tokens, word_ranks))
            else:
                outputs.append((sent_log_probs, sent_tokens))
            # output = (sent_log_probs.sum(), sent_ids, sent_tokens)
            # outputs.append(output)
        return outputs


class MambaScorer(LMScorer):
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        device: Optional[str] = "cuda",
        tokenizer=None,
        dtype=torch.float16,
        **kwargs,
    ) -> None:
        super(MambaScorer, self).__init__(model, device=device, tokenizer=tokenizer)
        if "cuda" not in device:
            raise Exception("SSM models are only supported through GPUs")

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        try:
            self.model = MambaLMHeadModel.from_pretrained(
                model, device=device, dtype=dtype
            )
            self.model.eval()
        except:
            raise Exception(
                "It seems you do not have mamba-ssm installed, please install it using `pip install mamba-ssm`, but remember that it only works with GPUs. If you do not have one, you cannot use a MambaScorer."
            )

        # define CLS and SEP tokens
        if self.tokenizer.pad_token is None:
            if tokenizer is not None:
                warnings.warn(
                    "tokenizer is changed by adding pad_token_id to the tokenizer."
                )
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": ["<pad>"]}
                )
                self.tokenizer.pad_token = "<pad>"
                self.model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.padding_side == "left":
            self.tokenizer.padding_side = "right"

        self.padding_side = self.tokenizer.padding_side

        try:
            self.bow_symbol = self.tokenizer.convert_ids_to_tokens(
                self.tokenizer(" ", add_special_tokens=False).input_ids[0]
            )[0]
        except:
            self.bow_symbol = None
        if (
            self.bow_symbol == self.tokenizer.bos_token
            or self.bow_symbol is None
            or self.bow_symbol == self.tokenizer.eos_token
        ):
            self.is_bow_tokenizer = False
        else:
            self.is_bow_tokenizer = True

        if self.is_bow_tokenizer:
            self.bow_subwords = defaultdict(lambda: False)
            for word, idx in self.tokenizer.get_vocab().items():
                if word[0] == self.bow_symbol:
                    self.bow_subwords[idx] = True
                else:
                    self.bow_subwords[idx] = False

            # for cases where the model has added tokens beyond the ones it comes with
            for idx, details in self.tokenizer.added_tokens_decoder.items():
                if details.lstrip == True:
                    self.bow_subwords[idx] = True

            self.bow_subwords = dict(self.bow_subwords)
            self.bow_subword_idx = [k for k, v in self.bow_subwords.items() if v]

    def encode(
        self,
        text: Union[str, List[str]],
        bos_token: bool = False,
        eos_token: bool = False,
    ) -> BatchEncoding:
        def _format(self, text, bos, eos):
            if bos:
                text = self.tokenizer.bos_token + text
            if eos:
                text = text + self.tokenizer.eos_token
            return text

        text = [text] if isinstance(text, str) else text
        text = [_format(self, t, bos_token, eos_token) for t in text]
        if self.tokenizer.chat_template is None:
            encoded = self.tokenizer(text, return_tensors="pt", padding=True)
        else:
            encoded = self.tokenizer(
                text, return_tensors="pt", padding=True, add_special_tokens=False
            )
        if "token_type_ids" in encoded.keys():
            encoded.pop("token_type_ids")

        return encoded

    def word_spans_tokenized(self, batch: Iterable, tokenize_function: Callable):
        """
        Aligns the spans of a string tokenized by a function to
        the tokenizer of the LM.

        :param batch: batch of sentences to be prepared for scoring.
        :param tokenize_function: the tokenizer function -- we recommend nltk.TweetTokenizer

        :return: Batch of index spans and the tokenized words.
        """
        all_spans = []
        all_lm_spans = []
        words = []
        for item in batch:
            # splitted = [tokenize_function(s) for s in item.split(" ")]
            if isinstance(tokenize_function, Callable):
                splitted = [tokenize_function(s) for s in item.split(" ")]
            else:
                splitted = tokenize_function[idx]
            words.append([" ".join(s) for s in splitted])
            spans = []
            lm_spans = []
            start = 0
            end = 0
            for i, entry in enumerate(splitted):
                end = len(entry)
                old_end = start + end
                spans.append((start, old_end))
                start += end

                joined = ' '.join(entry)
                if i != 0:
                    joined = f' {joined}'

                # mapping from tokenizer spans to LM tokenizer ids
                lm_tokenized = self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer(joined, add_special_tokens=False)[
                        "input_ids"
                    ]
                )
                len_diff = len(lm_tokenized) - len(entry)
                if i == 0:
                    new_start = 0
                else:
                    new_start = lm_spans[i - 1][-1]

                if len_diff > 0:
                    new_end = new_start + 1 + len_diff
                else:
                    new_end = new_start + 1
                lm_spans.append((new_start, new_end))
            all_spans.append(spans)
            all_lm_spans.append(lm_spans)
            # words.append([s[0] for s in splitted])

        return all_lm_spans, words

    def prepare_text(
        self,
        text: Union[str, List[str], BatchEncoding],
        bos_token: bool = False,
        eos_token: bool = False,
    ) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on.

        :param text: batch of sentences to be prepared for scoring.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        if isinstance(text, BatchEncoding):
            encoded = text
        else:
            encoded = self.encode(text, bos_token, eos_token)
        offsets = [0] * len(encoded["input_ids"])
        return encoded, offsets

    def prime_text(
        self,
        preamble: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        separator: Union[str, List[str]] = " ",
        bos_token=False,
        eos_token=False,
    ) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on.

        :param ``Union[str, List[str]]`` preamble: Batch of prefixes/prime/preambles on which the LM is conditioned.
        :param ``Union[str, List[str]]`` stimuli: Batch of continuations that are scored based on the conditioned text (provided in the ``preamble``). The positions of the elements match their counterparts in the ``preamble``.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        preamble_text = [preamble] if isinstance(preamble, str) else preamble
        preamble_encoded = self.tokenizer(preamble_text)["input_ids"]
        preamble_lens = []
        for preamble_tokens in preamble_encoded:
            if bos_token:
                restricted_id = float("inf")
                bos_offset = 1
            else:
                restricted_id = self.tokenizer.pad_token_id
                bos_offset = 0
            preamble_lens.append(
                len([token for token in preamble_tokens if token != restricted_id])
                - 1
                + bos_offset
            )

        if isinstance(separator, str):
            sentences = (
                [preamble + separator + stimuli]
                if isinstance(preamble, str)
                else [p + separator + s for p, s in list(zip(preamble, stimuli))]
            )

        elif isinstance(separator, list):
            assert not isinstance(preamble, str)
            assert len(preamble) == len(separator) == len(stimuli)

            sentences = [
                p + sep + s for p, sep, s in list(zip(preamble, separator, stimuli))
            ]

        return self.encode(sentences, bos_token, eos_token), preamble_lens

    def distribution(self, batch: Iterable) -> torch.Tensor:
        """
        Returns a distribution over the vocabulary of the model.

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.

        :return: Tensor consisting of log probabilies over vocab items.
        """
        batch, offsets = batch
        ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        if self.device != "auto":
            ids = ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids)
            logits = outputs.logits
            if "cuda" in self.device or "auto" in self.device:
                logits.detach()

        outputs = []
        for sent_index in range(len(ids)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(batch.tokens(sent_index))
                if sent_nopad_mask[i] and i > offsets[sent_index] + 1
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            # ignore first token (<|eos|>)
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            # sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")

            outputs.append(sent_logits[-1])
        return torch.stack(outputs, 0)

    def next_word_distribution(
        self, queries: List, bos_token=False, eos_token=False, surprisal: bool = False
    ):
        """
        Returns the log probability distribution of the next word.
        """
        encoded = self.encode(queries, bos_token, eos_token)
        if self.device != "auto":
            encoded = encoded.to(self.device)
        query_ids = [
            [j for j, i in enumerate(instance) if i != self.tokenizer.pad_token_id][-1]
            for instance in encoded["input_ids"].tolist()
        ]

        logits = self.model(encoded.input_ids).logits.detach()
        # logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits[torch.arange(len(query_ids)), query_ids]
        logprobs = logits - logits.logsumexp(1).unsqueeze(1)

        if surprisal:
            logprobs = -1.0 * logprobs

        return logprobs

    def compute_stats(
        self,
        batch: Iterable,
        rank: bool = False,
        prob: bool = False,
        base_two: bool = False,
        return_tensors: bool = False,
        bow_correction: bool = False,
    ) -> Union[Tuple[List[float], List[float]], List[float]]:
        """
        Primary computational method that processes a batch of prepared sentences and returns per-token scores for each sentence. By default, returns log-probabilities.

        :param ``Iterable`` batch: batched input as processed by ``prepare_text`` or ``prime_text``.
        :param ``bool`` rank: whether the model should also return ranks per word (based on the conditional log-probability of the word in context).
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param ``bool`` return_tensors: whether the model should return scores as a list of tensors instead of a list of lists. This is important in some other convenient methods used in the package.
        :param ``bool'' bow_correction: whether to apply the beginning of word correction, as pointed out in Pimentel and Meister (2024) and Oh and Schuler (2024).

        :return: Either a tuple of lists, each containing probabilities and ranks per token in each sentence passed in the input.
        :rtype: ``Union[Tuple[List[float], List[int]], List[float]]``
        """
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        encoded, offsets = batch
        if self.device != "auto":
            encoded = encoded.to(self.device)

        # ids = [
        #     [i for i in instance if i != self.tokenizer.pad_token_id]
        #     for instance in encoded["input_ids"].tolist()
        # ]
        ids = [
            [i for i, am in zip(instance, attention_mask) if am != 0]
            for instance, attention_mask in zip(
                encoded["input_ids"].tolist(), encoded["attention_mask"].tolist()
            )
        ]

        ## Ignore the probabilities of the first token.
        effective_ids = [id[1:] for id in ids]

        with torch.no_grad():
            logits = self.model(encoded.input_ids).logits.detach()

        # logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits.split([1] * len(offsets))

        ## Set up storage variables
        scores = []
        if rank:
            ranks = []

        for logit, idx, offset in zip(logits, effective_ids, offsets):
            length = len(idx)
            # logit = logit.squeeze(0)[torch.arange(offset, length),]

            # logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)
            query_ids = idx[offset:]
            logit = logit.squeeze(0)
            logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)

            actual_logprob_distribution = logprob_distribution[
                torch.arange(offset, length),
            ]

            score = actual_logprob_distribution[
                torch.arange(length - offset), query_ids
            ]

            if not self.is_bow_tokenizer:
                bow_correction = False

            if bow_correction:
                # mask_forward = torch.zeros_like(logprob_distribution)
                # mask_current = torch.zeros_like(logprob_distribution)
                mask_forward = torch.zeros(length).to(self.device)
                mask_current = torch.zeros(length).to(self.device)

                for i in range(len(idx)):
                    if i == len(idx) - 1:
                        mask_forward[i] = 1
                        if not self.bow_subwords[idx[i]]:
                            mask_current[i] = 0
                        else:
                            mask_current[i] = 1
                        break
                    elif self.bow_subwords[idx[i + 1]]:
                        mask_forward[i] = 1
                        if not self.bow_subwords[idx[i]]:
                            mask_current[i] = 0
                        else:
                            mask_current[i] = 1
                    else:
                        mask_forward[i] = 0
                        mask_current[i] = 1

                mask_forward = mask_forward[offset:]
                # mask_current = mask_current[offset:]
                mask_current = torch.roll(mask_forward, shifts=1)
                mask_current[0] = 0.0

                bow_subword_idx_tensor = torch.tensor(self.bow_subword_idx).to(
                    self.device
                )

                forward_correction = (
                    logprob_distribution[offset:][torch.arange(length - offset) + 1,]
                    .index_select(-1, bow_subword_idx_tensor)
                    .logsumexp(1)
                )

                current_correction = (
                    actual_logprob_distribution[torch.arange(length - offset),]
                    .index_select(-1, bow_subword_idx_tensor)
                    .logsumexp(1)
                )

                score = (
                    score
                    + (forward_correction * mask_forward)
                    - (current_correction * mask_current)
                )

            if base_two:
                """
                Log_2(X) = log_e(X)/log_e(2) (broadcasted)
                """
                score = score / torch.tensor(2).log()
            else:
                if prob:
                    score = score.exp()
                else:
                    score = score

            if rank:
                # shape = logprob_distribution.shape
                """
                Double argsort trick:
                first argsort returns idxes of values that would return a sorted tensor,
                second argsort returns ranks (0 indexed)

                Proof: https://www.berkayantmen.com/rank.html

                TODO: Try to implement ranking in linear time but across arbitrary dimensions:
                https://stackoverflow.com/a/5284703
                """
                word_ranks = (
                    -1.0 * actual_logprob_distribution
                ).argsort().argsort() + 1
                # inv_ranks = logprob_distribution.argsort().argsort() + 1
                # word_ranks = shape[1] - inv_ranks + 1
                word_ranks = word_ranks[
                    torch.arange(length - offset), query_ids
                ].tolist()
                ranks.append(word_ranks)

            scores.append(score)

        if not return_tensors:
            # scores = [torch.tensor(l).detach() for l in scores]
            scores = [s.tolist() for s in scores]

        if rank:
            return scores, ranks
        else:
            return scores

    def word_score_tokenized(
        self,
        batch: Iterable,
        tokenize_function: Callable,
        bow_correction: bool = False,
        bos_token: bool = False,
        eos_token: bool = False,
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        return_tensors: bool = False,
    ):
        """
        Returns the logprobs per word, as tokenized by the tokenize_function.

        :param batch: batch of sentences to be prepared for scoring.
        :param tokenize_function: tokenize function to maps strings to tokenized words.
        :param bow_correction: whether to apply Oh and Schuler's correction.
        :param bos_token: if a beginning of sentence token should be added (specify this as True for GPT2 and Pythia, for other they do it by default).
        :param eos_token: if an end of sentence token should be added
        :param prob: if the scores should be probabilities instead of logprobabilities
        :param base_two: if the logprobabilities should

        :return: Batch of words and their corresponding log-probs (summed log-probs of the tokens)
        """
        batch = [batch] if isinstance(batch, str) else batch

        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        tokenized = [" ".join(tokenize_function(item)) for item in batch]

        encoded, offsets = self.prepare_text(
            batch, bos_token=bos_token, eos_token=eos_token
        )

        scores = self.compute_stats(
            (encoded, offsets),
            bow_correction=bow_correction,
            prob=prob,
            base_two=base_two,
            return_tensors=True,
        )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        spans, words = self.word_spans_tokenized(tokenized, tokenize_function)

        word_scores = []
        for i, span in enumerate(spans):
            ws = []
            for j, (s, e) in enumerate(span):
                if prob:
                    score = scores[i][s:e].prod()
                else:
                    score = scores[i][s:e].sum()
                if not return_tensors:
                    score = score.item()

                ws.append((words[i][j], score))
            word_scores.append(ws)

        return word_scores

    def conditional_word_score_tokenized(
        self,
        prefix,
        stimuli,
        tokenize_function: Callable,
        separator=" ",
        bow_correction=False,
        bos_token=False,
        eos_token=False,
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        return_tensors: bool = False,
    ):

        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        prefix = [prefix] if isinstance(prefix, str) else prefix
        stimuli = [stimuli] if isinstance(stimuli, str) else stimuli

        # tokenized = [" ".join(tokenize_function(item)) for item in stimuli]
        if isinstance(tokenize_function, Callable):
            tokenized = [" ".join(tokenize_function(item)) for item in stimuli]
        elif isinstance(tokenize_function, list):
            tokenize_function = [[[tt] for tt in t] for t in tokenize_function]
            tokenized = [" ".join([tt[0] for tt in t]) for t in tokenize_function]
        else:
            raise ValueError("Incorrect type of tokenize_function argument (either callable or list)")

        encoded, offsets = self.prime_text(
            prefix,
            stimuli,
            separator=separator,
            bos_token=bos_token,
            eos_token=eos_token,
        )

        scores = self.compute_stats(
            (encoded, offsets),
            bow_correction=bow_correction,
            prob=prob,
            base_two=base_two,
            return_tensors=True,
        )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        spans, words = self.word_spans_tokenized(tokenized, tokenize_function)

        word_scores = []
        for i, span in enumerate(spans):
            ws = []
            for j, (s, e) in enumerate(span):
                if prob:
                    score = scores[i][s:e].prod()
                else:
                    score = scores[i][s:e].sum()
                if not return_tensors:
                    score = score.item()

                ws.append((words[i][j], score))
            word_scores.append(ws)

        return word_scores

    def sequence_score(
        self,
        batch,
        reduction=lambda x: x.mean(0).item(),
        base_two=False,
        bow_correction=False,
        **kwargs,
    ):
        """
        TODO: reduction should be a string, if it's a function, specify what kind of function. --> how to ensure it is always that type?
        """
        tokenized = self.prepare_text(batch, **kwargs)
        scores = self.compute_stats(
            tokenized,
            rank=False,
            base_two=base_two,
            bow_correction=bow_correction,
            return_tensors=True,
        )
        reduced = list(map(reduction, scores))
        return reduced

    def token_score(
        self,
        batch: Union[str, List[str]],
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        rank: bool = False,
        decode: bool = False,
        bow_correction: bool = False,
        **kwargs,
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        """
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)
        :param ``bool'' bow_correction: whether to apply the beginning of word correction, as pointed out in Pimentel and Meister (2024) and Oh and Schuler (2024).

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        """

        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        tokenized = self.prepare_text(batch, **kwargs)
        if rank:
            scores, ranks = self.compute_stats(
                tokenized,
                rank=rank,
                prob=prob,
                base_two=base_two,
                bow_correction=bow_correction,
                return_tensors=True,
            )
        else:
            scores = self.compute_stats(
                tokenized,
                prob=prob,
                base_two=base_two,
                bow_correction=bow_correction,
                return_tensors=True,
            )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        scores = [s.tolist() for s in scores]

        # indices = [
        #     [i for i in indexed if i != self.tokenizer.pad_token_id]
        #     for indexed in tokenized[0]["input_ids"].tolist()
        # ]

        indices = [
            [i for i, am in zip(instance, attention_mask) if am != 0]
            for instance, attention_mask in zip(
                tokenized[0]["input_ids"].tolist(),
                tokenized[0]["attention_mask"].tolist(),
            )
        ]
        if decode:
            tokens = [self.decode(idx) for idx in indices]
        else:
            tokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in indices]

        if rank:
            assert len(tokens) == len(scores) == len(ranks)
        else:
            assert len(tokens) == len(scores)

        res = []
        if rank:
            for t, s, r in zip(tokens, scores, ranks):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    ra = [0] * diff + r
                    res.append(list(zip(t, sc, ra)))
                else:
                    res.append(list(zip(t, sc, ra)))
            # return [list(zip(t, s, r)) for t, s, r in zip(tokens, scores, ranks)]
        else:
            for t, s in zip(tokens, scores):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    res.append(list(zip(t, sc)))
                else:
                    res.append(list(zip(t, sc)))

        return res


class VLMScorer(LMScorer):
    def __init__(self, model, device, tokenizer=None, causallm=False, **kwargs):
        super(VLMScorer, self).__init__(model, device=device, tokenizer=tokenizer)
        self.causallm = causallm
        if isinstance(model, str):
            if self.causallm:
                self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
                self.processor = AutoProcessor.from_pretrained(model, **kwargs)
                if self.device == "auto":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model, device_map=self.device, **kwargs
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
            else:
                self.tokenizer = AutoProcessor.from_pretrained(model, **kwargs)
                if self.device == "auto":
                    try:
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            model, device_map=self.device, **kwargs
                        )
                    except:
                        self.model = AutoModelForVision2Seq.from_pretrained(
                            model, device_map=self.device, **kwargs
                        )
                else:
                    try:
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            model, **kwargs
                        )
                    except:
                        self.model = AutoModelForVision2Seq.from_pretrained(model, **kwargs)
        else:
            self.model = model

        if self.device != "auto":
            self.model.to(self.device)

        if "processor" in type(self.tokenizer).__name__.lower():
            if self.tokenizer.tokenizer.padding_side == "left":
                self.tokenizer.tokenizer.padding_side = "right"
            self.padding_side = self.tokenizer.tokenizer.padding_side
            self.pad_token_id = self.tokenizer.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.tokenizer.eos_token_id
            self.bos_token_id = self.tokenizer.tokenizer.bos_token_id
            try:
                self.bow_symbol = self.tokenizer.tokenizer.convert_ids_to_tokens(
                    self.tokenizer.tokenizer(" ", add_special_tokens=False).input_ids[0]
                )[0]
            except:
                self.bow_symbol = None

            if (
                self.bow_symbol == self.tokenizer.tokenizer.bos_token
                or self.bow_symbol is None
                or self.bow_symbol == self.tokenizer.tokenizer.eos_token
            ):
                self.is_bow_tokenizer = False
            else:
                self.is_bow_tokenizer = True

            if self.is_bow_tokenizer:
                self.bow_subwords = defaultdict(lambda: False)
                for word, idx in self.tokenizer.tokenizer.get_vocab().items():
                    if word[0] == self.bow_symbol:
                        self.bow_subwords[idx] = True
                    else:
                        self.bow_subwords[idx] = False

                self.bow_subwords = dict(self.bow_subwords)
                self.bow_subword_idx = [k for k, v in self.bow_subwords.items() if v]
        else:
            if self.tokenizer.padding_side == "left":
                self.tokenizer.padding_side = "right"
            self.padding_side = self.tokenizer.padding_side
            self.pad_token_id = self.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.bos_token_id = self.tokenizer.bos_token_id

            try:
                self.bow_symbol = self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer(" ", add_special_tokens=False).input_ids[0]
                )
            except:
                self.bow_symbol = None

            if (
                self.bow_symbol == self.tokenizer.bos_token
                or self.bow_symbol is None
                or self.bow_symbol == self.tokenizer.eos_token
            ):
                self.is_bow_tokenizer = False
            else:
                self.is_bow_tokenizer = True

            if self.is_bow_tokenizer:
                self.bow_subwords = defaultdict(lambda: False)
                for word, idx in self.tokenizer.get_vocab().items():
                    if word[0] == self.bow_symbol:
                        self.bow_subwords[idx] = True
                    else:
                        self.bow_subwords[idx] = False

                # for cases where the model has added tokens beyond the ones it comes with
                for idx, details in self.tokenizer.added_tokens_decoder.items():
                    if details.lstrip == True:
                        self.bow_subwords[idx] = True

                self.bow_subwords = dict(self.bow_subwords)
                self.bow_subword_idx = [k for k, v in self.bow_subwords.items() if v]

        if isinstance(model, str):
            self.model.eval()

        # try:
        #     self.bow_symbol = self.tokenizer.convert_ids_to_tokens(
        #         self.tokenizer(" ", add_special_tokens=False).input_ids[0]
        #     )
        # except:
        #     self.bow_symbol = None

        # if (
        #     self.bow_symbol == self.tokenizer.bos_token
        #     or self.bow_symbol is None
        #     or self.bow_symbol == self.tokenizer.eos_token
        # ):
        #     self.is_bow_tokenizer = False
        # else:
        #     self.is_bow_tokenizer = True

        # if self.is_bow_tokenizer:
        #     self.bow_subwords = defaultdict(lambda: False)
        #     for word, idx in self.tokenizer.get_vocab().items():
        #         if word[0] == self.bow_symbol:
        #             self.bow_subwords[idx] = True
        #         else:
        #             self.bow_subwords[idx] = False

        #     self.bow_subwords = dict(self.bow_subwords)
        #     self.bow_subword_idx = [k for k, v in self.bow_subwords.items() if v]

    def encode(self, text: Union[str, List[str]], image=None) -> BatchEncoding:
        # def _format(self, text, image, bos, eos):
        #     if bos:
        #         text = self.tokenizer.bos_token + text
        #     if eos:
        #         text = text + self.tokenizer.eos_token
        #     if image is None:
        #         img = ""
        #     else:
        #         img = f"<image>\n"

        #     return img + text

        text = [text] if isinstance(text, str) else text
        image = [image] if isinstance(image, Image.Image) else image
        # text = [_format(self, t, image, bos_token, eos_token) for t in text]
        if self.causallm:
            encoded_text = self.tokenizer(text, return_tensors="pt", padding=True)
            encoded_images = self.processor(images=image, return_tensors="pt")
            encoded = BatchEncoding(
                {k: v for obj in [encoded_text, encoded_images] for k, v in obj.items()}
            )
        else:
            encoded = self.tokenizer(
                text=text, images=image, return_tensors="pt", padding=True
            )
        if "token_type_ids" in encoded.keys():
            encoded.pop("token_type_ids")

        return encoded

    def decode(self, idx: List[int]):
        """
        Decode input ids using the model's tokenizer.

        :param ``List[int]`` idx: List of ids.

        :return: Decoded strings
        :rtype: List[str]
        """
        if "processor" in type(self.tokenizer).__name__.lower():
            decoded = [
                self.tokenizer.decode([x]).strip()
                for x in self.tokenizer.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenizer.convert_ids_to_tokens(idx)
                )
            ]
        else:
            decoded = [
                self.tokenizer.decode([x]).strip()
                for x in self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.convert_ids_to_tokens(idx)
                )
            ]
        return decoded

    def prepare_text(
        self, text: Union[str, List[str], BatchEncoding], image=None
    ) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on.

        :param text: batch of sentences to be prepared for scoring.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        if isinstance(text, BatchEncoding):
            encoded = text
        else:
            encoded = self.encode(text, image)
        offsets = [0] * len(encoded["input_ids"])
        return encoded, offsets

    def prime_text(
        self,
        preamble: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        image=None,
        separator: Union[str, List[str]] = " ",
    ) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on.

        :param ``Union[str, List[str]]`` preamble: Batch of prefixes/prime/preambles on which the LM is conditioned.
        :param ``Union[str, List[str]]`` stimuli: Batch of continuations that are scored based on the conditioned text (provided in the ``preamble``). The positions of the elements match their counterparts in the ``preamble``.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        preamble_text = [preamble] if isinstance(preamble, str) else preamble
        preamble_encoded = self.tokenizer(
            text=preamble_text, images=image, padding=True
        )["input_ids"]
        preamble_lens = []
        for preamble_tokens in preamble_encoded:
            restricted_id = self.pad_token_id
            bos_offset = 0
            preamble_lens.append(
                len([token for token in preamble_tokens if token != restricted_id])
                - 1
                + bos_offset
            )

        if isinstance(separator, str):
            sentences = (
                [preamble + separator + stimuli]
                if isinstance(preamble, str)
                else [p + separator + s for p, s in list(zip(preamble, stimuli))]
            )

        elif isinstance(separator, list):
            assert not isinstance(preamble, str)
            assert len(preamble) == len(separator) == len(stimuli)

            sentences = [
                p + sep + s for p, sep, s in list(zip(preamble, separator, stimuli))
            ]

        return self.encode(sentences, image), preamble_lens

    def next_word_distribution(
        self, queries: List, image: List, surprisal: bool = False
    ):
        """
        Returns the log probability distribution of the next word.
        """
        encoded = self.encode(queries, image)
        if self.device != "auto":
            encoded = encoded.to(self.device)
        query_ids = [
            [j for j, i in enumerate(instance) if i != self.pad_token_id][-1]
            for instance in encoded["input_ids"].tolist()
        ]

        logits = self.model(**encoded).logits.detach()
        # logits[:, :, self.pad_token_id] = float("-inf")

        logits = logits[torch.arange(len(query_ids)), query_ids]
        logprobs = logits - logits.logsumexp(1).unsqueeze(1)

        if surprisal:
            logprobs = -1.0 * logprobs

        return logprobs

    def compute_stats(
        self,
        batch: Iterable,
        rank: bool = False,
        prob: bool = False,
        base_two: bool = False,
        return_tensors: bool = False,
        bow_correction: bool = False,
    ) -> Union[Tuple[List[float], List[float]], List[float]]:
        """
        Primary computational method that processes a batch of prepared sentences and returns per-token scores for each sentence. By default, returns log-probabilities.

        :param ``Iterable`` batch: batched input as processed by ``prepare_text`` or ``prime_text``.
        :param ``bool`` rank: whether the model should also return ranks per word (based on the conditional log-probability of the word in context).
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param ``bool`` return_tensors: whether the model should return scores as a list of tensors instead of a list of lists. This is important in some other convenient methods used in the package.
        :param ``bool'' bow_correction: whether to apply the beginning of word correction, as pointed out in Pimentel and Meister (2024) and Oh and Schuler (2024).

        :return: Either a tuple of lists, each containing probabilities and ranks per token in each sentence passed in the input.
        :rtype: ``Union[Tuple[List[float], List[int]], List[float]]``
        """
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        encoded, offsets = batch
        cutoff = encoded["input_ids"].shape[-1]
        if self.device != "auto":
            encoded = encoded.to(self.device)

        # ids = [
        #     [i for i in instance if i != self.tokenizer.pad_token_id]
        #     for instance in encoded["input_ids"].tolist()
        # ]
        ids = [
            [i for i, am in zip(instance, attention_mask) if am != 0]
            for instance, attention_mask in zip(
                encoded["input_ids"].tolist(), encoded["attention_mask"].tolist()
            )
        ]

        ## Ignore the probabilities of the first token.
        effective_ids = [id[1:] for id in ids]

        with torch.no_grad():
            logits = self.model(**encoded).logits.detach()[:, -cutoff:, :]

        # logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits.split([1] * len(offsets))

        ## Set up storage variables
        scores = []
        if rank:
            ranks = []

        for logit, idx, offset in zip(logits, effective_ids, offsets):
            length = len(idx)
            # logit = logit.squeeze(0)[torch.arange(offset, length),]

            # logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)
            query_ids = idx[offset:]
            logit = logit.squeeze(0)
            logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)

            actual_logprob_distribution = logprob_distribution[
                torch.arange(offset, length),
            ]

            score = actual_logprob_distribution[
                torch.arange(length - offset), query_ids
            ]

            if not self.is_bow_tokenizer:
                bow_correction = False

            if bow_correction:
                # mask_forward = torch.zeros_like(logprob_distribution)
                # mask_current = torch.zeros_like(logprob_distribution)
                mask_forward = torch.zeros(length).to(self.device)
                mask_current = torch.zeros(length).to(self.device)

                for i in range(len(idx)):
                    if i == len(idx) - 1:
                        mask_forward[i] = 1
                        if not self.bow_subwords[idx[i]]:
                            mask_current[i] = 0
                        else:
                            mask_current[i] = 1
                        break
                    elif self.bow_subwords[idx[i + 1]]:
                        mask_forward[i] = 1
                        if not self.bow_subwords[idx[i]]:
                            mask_current[i] = 0
                        else:
                            mask_current[i] = 1
                    else:
                        mask_forward[i] = 0
                        mask_current[i] = 1

                mask_forward = mask_forward[offset:]
                # mask_current = mask_current[offset:]
                mask_current = torch.roll(mask_forward, shifts=1)
                mask_current[0] = 0.0

                bow_subword_idx_tensor = torch.tensor(self.bow_subword_idx).to(
                    self.device
                )

                forward_correction = (
                    logprob_distribution[offset:][torch.arange(length - offset) + 1,]
                    .index_select(-1, bow_subword_idx_tensor)
                    .logsumexp(1)
                )

                current_correction = (
                    actual_logprob_distribution[torch.arange(length - offset),]
                    .index_select(-1, bow_subword_idx_tensor)
                    .logsumexp(1)
                )

                score = (
                    score
                    + (forward_correction * mask_forward)
                    - (current_correction * mask_current)
                )

            if base_two:
                """
                Log_2(X) = log_e(X)/log_e(2) (broadcasted)
                """
                score = score / torch.tensor(2).log()
            else:
                if prob:
                    score = score.exp()
                else:
                    score = score

            if rank:
                # shape = logprob_distribution.shape
                """
                Double argsort trick:
                first argsort returns idxes of values that would return a sorted tensor,
                second argsort returns ranks (0 indexed)

                Proof: https://www.berkayantmen.com/rank.html

                TODO: Try to implement ranking in linear time but across arbitrary dimensions:
                https://stackoverflow.com/a/5284703
                """
                word_ranks = (
                    -1.0 * actual_logprob_distribution
                ).argsort().argsort() + 1
                # inv_ranks = logprob_distribution.argsort().argsort() + 1
                # word_ranks = shape[1] - inv_ranks + 1
                word_ranks = word_ranks[
                    torch.arange(length - offset), query_ids
                ].tolist()
                ranks.append(word_ranks)

            scores.append(score)

        if not return_tensors:
            # scores = [torch.tensor(l).detach() for l in scores]
            scores = [s.tolist() for s in scores]

        if rank:
            return scores, ranks
        else:
            return scores

    def sequence_score(
        self,
        text_batch,
        image_batch=None,
        reduction=lambda x: x.mean(0).item(),
        base_two=False,
        bow_correction: bool = False,
    ):
        """
        TODO: reduction should be a string, if it's a function, specify what kind of function. --> how to ensure it is always that type?
        """
        tokenized = self.prepare_text(text_batch, image_batch)
        scores = self.compute_stats(
            tokenized,
            rank=False,
            base_two=base_two,
            bow_correction=bow_correction,
            return_tensors=True,
        )
        reduced = list(map(reduction, scores))
        return reduced

    def token_score(
        self,
        text_batch: Union[str, List[str]],
        image_batch=None,
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        rank: bool = False,
        decode: bool = True,
        bow_correction: bool = False,
        **kwargs,
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        """
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)
        :param ``bool'' bow_correction: whether to apply the beginning of word correction, as pointed out in Pimentel and Meister (2024) and Oh and Schuler (2024).

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        """

        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        tokenized = self.prepare_text(text_batch, image_batch, **kwargs)
        if rank:
            scores, ranks = self.compute_stats(
                tokenized,
                rank=rank,
                prob=prob,
                base_two=base_two,
                bow_correction=bow_correction,
                return_tensors=True,
            )
        else:
            scores = self.compute_stats(
                tokenized,
                prob=prob,
                base_two=base_two,
                bow_correction=bow_correction,
                return_tensors=True,
            )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        scores = [s.tolist() for s in scores]

        # indices = [
        #     [i for i in indexed if i != self.tokenizer.pad_token_id]
        #     for indexed in tokenized[0]["input_ids"].tolist()
        # ]

        indices = [
            [i for i, am in zip(instance, attention_mask) if am != 0]
            for instance, attention_mask in zip(
                tokenized[0]["input_ids"].tolist(),
                tokenized[0]["attention_mask"].tolist(),
            )
        ]
        if decode:
            tokens = [self.decode(idx) for idx in indices]
        else:
            if "processor" in type(self.tokenizer).__name__.lower():
                tokens = [
                    self.tokenizer.tokenizer.convert_ids_to_tokens(idx)
                    for idx in indices
                ]
            else:
                tokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in indices]

        if rank:
            assert len(tokens) == len(scores) == len(ranks)
        else:
            assert len(tokens) == len(scores)

        res = []
        if rank:
            for t, s, r in zip(tokens, scores, ranks):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    ra = [0] * diff + r
                    res.append(list(zip(t, sc, ra)))
                else:
                    res.append(list(zip(t, sc, ra)))
            # return [list(zip(t, s, r)) for t, s, r in zip(tokens, scores, ranks)]
        else:
            for t, s in zip(tokens, scores):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    res.append(list(zip(t, sc)))
                else:
                    res.append(list(zip(t, sc)))

        return res

    def conditional_score(
        self,
        prefix: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        image=None,
        separator: str = " ",
        reduction: Callable = lambda x: x.mean(0).item(),
        prob: bool = False,
        base_two: bool = False,
        bow_correction: bool = False,
        **kw,
    ) -> List[float]:
        """
        Pooled estimates of sequence log probabilities (or some modification of it), given a prefix. Pooling is usually done using a function that is passed to the method.

        :param prefix: a batch of prefixes or primes passed to the
            language model. This is what the sequence is conditioned on, and the model ignores the word probabilities of this part of the input in estimating the overall score.
        :type prefix: ``Union[str, List[str]]``
        :param stimuli: a batch of sequences (same length as prefix)
            that form the main input consisting of the sequence whose
            score you want to calculate.
        :type stimuli: ``Union[str, List[str]]``
        :param reduction: Reduction function, is selected to be
            ``lambda x: x.mean(0).item()`` by default, which stands for the avg. log-probability per token for each sequence in the batch.
        :type reduction: Callable
        :param ``bool'' bow_correction: whether to apply the beginning of word correction, as pointed out in Pimentel and Meister (2024) and Oh and Schuler (2024).
        :param kw: model-specific keyword arguments to pass to the `prepare_text` function
        :return: List of floats specifying the desired score for the stimuli part of the input, e.g., P(stimuli | preamble).
        :rtype: ``List[float]``
        """
        primed = self.prime_text(prefix, stimuli, image, separator, **kw)

        result = self.compute_stats(
            primed,
            rank=False,
            base_two=base_two,
            prob=prob,
            bow_correction=bow_correction,
            return_tensors=True,
        )
        logprob = result
        reduced = list(map(reduction, logprob))

        return reduced
