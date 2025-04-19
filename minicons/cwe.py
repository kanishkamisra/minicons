"""Utilities to extract contextual word embeddings from Language Models."""

from PIL import Image

from typing import Union, List, Optional, Tuple

from .utils import (
    find_pattern,
    character_span,
    leading_whitespace_behavior,
    index_states,
)

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.utils.logging import set_verbosity_error
import torch

set_verbosity_error()


class CWE(object):
    """
    Implements the contextualized word embedding class to
    facilitate extraction of word representations from a given
    transformer model. [TEST]

    :param model_name: name of the model, should either be a path
        to a model (.pt or .bin file) stored locally, or a
        pretrained model stored on the Huggingface Model Hub.
    :type model_name: str
    :param device: device type that the model should be loaded on,
        options: `cpu or cuda:{0, 1, ...}`
    :type device: str, optional
    :param pretrained: whether to load the model with pretrained weights.
        loads a randomly initialized model if `False`. Default = `True`.
    :type pretrained: bool
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = "cpu",
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        """
        :param model_name: name of the model, should either be a path
            to a model (.pt or .bin file) stored locally, or a
            pretrained model stored on the Huggingface Model Hub.
        :type model_name: str
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        :param pretrained: whether to load the model with pretrained weights.
            loads a randomly initialized model if `False`. Default = `True`.
        :type pretrained: bool
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, **kwargs
        )
        self.pretrained = pretrained
        self.model_name = model_name
        if pretrained:
            self.model = AutoModel.from_pretrained(
                model_name, return_dict=True, output_hidden_states=True, **kwargs
            )
        else:
            self.config = AutoConfig.from_pretrained(
                model_name, return_dict=True, output_hidden_states=True, **kwargs
            )
            self.model = AutoModel.from_config(self.config)
        self.dimensions = list(self.model.parameters())[-1].shape[0]

        self.layers = self.model.config.num_hidden_layers

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|pad|>"]}
            )
            self.tokenizer.pad_token = "<|pad|>"
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.leading_whitespace_behavior = leading_whitespace_behavior(self.tokenizer)

        if self.processor.tokenizer.padding_side == "left":
            self.processor.tokenizer.padding_side = "right"

        self.model.to(self.device)
        self.model.eval()

    def encode_text(
        self, text: Union[str, List[str]], layer: Union[int, List[int]] = None
    ) -> Tuple:
        """
        Encodes batch of raw sentences using the model to return hidden
        states at a given layer.

        :param ``Union[str, List[str]]`` text: batch of raw sentences
        :param layer: layer from which the hidden states are extracted.
        :type layer: int
        :return: Tuple `(input_ids, hidden_states)`
        """
        sentences = [text] if isinstance(text, str) else text

        # Encode sentence into ids stored in the model's embedding layer(s).
        encoded = self.tokenizer(sentences, padding="longest", return_tensors="pt")
        encoded = encoded.to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        if "cuda" in self.device:
            attention_mask = attention_mask.cpu()

        if layer == "static" or layer == "pre":
            attention_mask = attention_mask.to(self.device)
            hidden_states = [
                self.model.get_output_embeddings().weight.detach()[i] for i in input_ids
            ]
            hidden_states = torch.stack(hidden_states)
            hidden_states = hidden_states * attention_mask
        else:
            # Compute hidden states for the sentence for the given layer.
            output = self.model(**encoded)

            # Hidden states appear as the last element of the otherwise custom hidden_states object
            if isinstance(layer, list) or layer == "all":
                hidden_states = output.hidden_states
                if "cuda" in self.device:
                    input_ids = input_ids.cpu()
                    hidden_states = [h.detach().cpu() for h in hidden_states]
                else:
                    hidden_states = [h.detach() for h in hidden_states]
                if layer != "all":
                    hidden_states = [hidden_states[i] for i in sorted(layer)]

                hidden_states = [h * attention_mask for h in hidden_states]
            else:
                # if layer != 'all':
                if layer is None:
                    layer = self.layers
                elif layer > self.layers:
                    raise ValueError(
                        f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!"
                    )
                hidden_states = output.hidden_states[layer]
                if "cuda" in self.device:
                    input_ids = input_ids.cpu()
                    hidden_states = hidden_states.detach().cpu()
                else:
                    hidden_states = hidden_states.detach()

                hidden_states = hidden_states * attention_mask
                # else:
                #     hidden_states = output.hidden_states

                #     if "cuda" in self.device:
                #         input_ids = input_ids.cpu()
                #         hidden_states = [h.detach().cpu() for h in hidden_states]
                #     else:
                #         hidden_states = [h.detach() for h in hidden_states]

        return input_ids, hidden_states

    # A function that extracts the representation of a given word in a sentence (first occurrence only)

    def extract_representation(
        self,
        sentence_words: Union[
            List[List[Union[str, Union[Tuple[int, int], str]]]],
            List[Union[str, Union[Tuple[int, int], str]]],
        ],
        layer: Union[int, List[int]] = None,
        multi_strategy="last",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Extract representations from the model at a given layer.

        :param ``Union[List[List[Union[str, Union[Tuple(int, int), str]]]], List[Union[str, Union[Tuple(int, int), str]]]]`` sentence_words: Input
            consisting of `[(sentence, word)]`, where sentence is an
            input sentence, and word is a word present in the sentence
            that will be masked out, or `[(sentence, (start, end))]`,
            where (start, end) is a tuple consisting of the character
            span indices that form the word.
        :param ``Union[int, List[int]]`` layer: layer(s) from which the hidden states are extracted.
        :return: torch tensors or list of torch
            tensors corresponding to word representations
        """
        sentences = (
            [sentence_words] if isinstance(sentence_words[0], str) else sentence_words
        )

        num_inputs = len(sentences)

        input_ids, hidden_states = self.encode_text(
            list(list(zip(*sentences))[0]), layer
        )

        if isinstance(sentences[0][1], str):
            sentences = [(s, character_span(s, w)) for s, w in sentences]

        search_queries = []
        for s, idx in sentences:
            if 0 in idx:
                search_queries.append(
                    self.tokenizer.encode_plus(
                        f"{s[idx[0]:idx[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif (
                self.leading_whitespace_behavior == "llama"
            ):  ## Seems like the GPT2 approach really fails w Llama's tokenizer
                search_queries.append(
                    self.tokenizer.encode_plus(
                        f"{s[idx[0]:idx[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif self.leading_whitespace_behavior == "gpt2":
                ## this one really matters if we are using GPT2
                search_queries.append(
                    self.tokenizer.encode_plus(
                        f" {s[idx[0]:idx[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif "mixed" in self.leading_whitespace_behavior:
                ## If behavior is mixed, let's stick with the majority of the tokenizer's behavior, but throw in a warning:
                print(
                    "Tokenizer has inconsistent behavior around leading whitespaces: this may lead to issues."
                )
                if self.leading_whitespace_behavior == "gpt2-mixed":
                    search_queries.append(
                        self.tokenizer.encode_plus(
                            f" {s[idx[0]:idx[1]]}", add_special_tokens=False
                        )["input_ids"]
                    )
                elif self.leading_whitespace_behavior == "llama-mixed":
                    search_queries.append(
                        self.tokenizer.encode_plus(
                            f"{s[idx[0]:idx[1]]}", add_special_tokens=False
                        )["input_ids"]
                    )

        query_idx = list(
            map(
                lambda x: find_pattern(x[0], x[1], multi_strategy),
                zip(search_queries, input_ids.tolist()),
            )
        )

        if multi_strategy in ["first", "last"]:

            query_idx = [q[0] for q in query_idx]

            if isinstance(layer, list) or layer == "all":
                representations = list(
                    map(
                        lambda x: torch.stack(
                            [
                                hs.squeeze()[idx[0] : idx[1]].mean(0)
                                for hs, idx in zip(x.split([1] * num_inputs), query_idx)
                            ]
                        ),
                        hidden_states,
                    )
                )
            else:
                if layer is None:
                    layer = self.layers
                elif layer == "static" or "pre":
                    layer = layer
                elif layer > self.layers:
                    raise ValueError(
                        f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!"
                    )
                # representations = hidden_states[torch.arange(num_inputs)[:, None], query_idx].mean(1)
                representations = torch.stack(
                    [
                        hs.squeeze()[idx[0] : idx[1]].mean(0)
                        for hs, idx in zip(
                            hidden_states.split([1] * num_inputs), query_idx
                        )
                    ]
                )
        else:
            if isinstance(layer, list) or layer == "all":
                representations = list(
                    map(lambda x: index_states(x, num_inputs, query_idx), hidden_states)
                )
            else:
                if layer is None:
                    layer = self.layers
                elif layer == "static" or "pre":
                    layer = layer
                elif layer > self.layers:
                    raise ValueError(
                        f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!"
                    )
                representations = index_states(hidden_states, num_inputs, query_idx)

        return representations

    def extract_sentence_representation(
        self,
        sentences: Union[str, List[str]],
        layer: Union[int, List[int]],
        pooler: str = "mean",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Extract representations of input sentences from one or more layers in the model.

        :param ``Union[str, List[str]]`` sentences: Input consisting of one or
            more sentences.
        :param ``Union[int, List[int]]`` layer: one or more layer from which the representations are
            extracted.
        :param pooler: pooling logic (mean, max, min)
        :type pooler: str

        :return: torch.Tensor
        """

        sentences = [sentences] if isinstance(sentences, str) else sentences

        input_ids, hidden_states = self.encode_text(sentences, layer)

        lengths = torch.tensor(
            [len([i for i in ids if i != 0]) for ids in input_ids.tolist()]
        )

        if isinstance(hidden_states, list):
            hidden_states = torch.stack(hidden_states).mean(0)

        if pooler == "mean":
            hidden_states = torch.div(hidden_states.sum(1), lengths.view(-1, 1))
        elif pooler == "min":
            hidden_states = hidden_states.min(0).values
        elif pooler == "max":
            hidden_states = hidden_states.max(0).values
        else:
            print("Only pooler = mean, max, or min supported for now!")

        return hidden_states

    def extract_paired_representations(
        self,
        sentence_words: Union[Tuple[str], List[Tuple[str]]],
        layer: int = None,
        multi_strategy="last",
    ) -> Tuple:
        """
        Extract representations of pairs of words from a given sentence
        from the model at a given layer.

        :param ``Union[Tuple[str], List[Tuple[str]]]`` text: Input
            consisting of `[(sentence, word1, word2)]`, where sentence
            is an input sentence, and word1, word2 are two words
            present in the sentence that will be masked out.
        :param layer: layer from which the representations are
            extracted.
        :type layer: int
        :return: Tuple consisting of torch tensors or lists of torch
            tensors corresponding to word representations
        """
        sentences = (
            [sentence_words] if isinstance(sentence_words[0], str) else sentence_words
        )

        num_inputs = len(sentences)

        input_ids, hidden_states = self.encode_text(
            list(list(zip(*sentences))[0]), layer
        )

        if isinstance(sentences[0][1], str):
            sentences = [
                (s, character_span(s, w1), character_span(s, w2))
                for s, w1, w2 in sentences
            ]

        search_queries1 = []
        search_queries2 = []
        for s, idx1, idx2 in sentences:
            if 0 in idx1:
                search_queries1.append(
                    self.tokenizer.encode_plus(
                        f"{s[idx1[0]:idx1[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif (
                self.leading_whitespace_behavior == "llama"
            ):  ## Seems like the GPT2 approach really fails w Llama's tokenizer
                search_queries1.append(
                    self.tokenizer.encode_plus(
                        f"{s[idx1[0]:idx1[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif self.leading_whitespace_behavior == "gpt2":
                ## this one really matters if we are using GPT2
                search_queries1.append(
                    self.tokenizer.encode_plus(
                        f" {s[idx1[0]:idx1[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif "mixed" in self.leading_whitespace_behavior:
                ## If behavior is mixed, let's stick with the majority of the tokenizer's behavior, but throw in a warning:
                print(
                    "Tokenizer has inconsistent behavior around leading whitespaces: this may lead to issues."
                )
                if self.leading_whitespace_behavior == "gpt2-mixed":
                    search_queries1.append(
                        self.tokenizer.encode_plus(
                            f" {s[idx1[0]:idx1[1]]}", add_special_tokens=False
                        )["input_ids"]
                    )
                elif self.leading_whitespace_behavior == "llama-mixed":
                    search_queries1.append(
                        self.tokenizer.encode_plus(
                            f"{s[idx1[0]:idx1[1]]}", add_special_tokens=False
                        )["input_ids"]
                    )
            if 0 in idx2:
                search_queries2.append(
                    self.tokenizer.encode_plus(
                        f"{s[idx2[0]:idx2[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif (
                self.leading_whitespace_behavior == "llama"
            ):  ## Seems like the GPT2 approach really fails w Llama's tokenizer
                search_queries2.append(
                    self.tokenizer.encode_plus(
                        f"{s[idx2[0]:idx2[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif self.leading_whitespace_behavior == "gpt2":
                ## this one really matters if we are using GPT2
                search_queries2.append(
                    self.tokenizer.encode_plus(
                        f" {s[idx2[0]:idx2[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif "mixed" in self.leading_whitespace_behavior:
                ## If behavior is mixed, let's stick with the majority of the tokenizer's behavior, but throw in a warning:
                print(
                    "Tokenizer has inconsistent behavior around leading whitespaces: this may lead to issues."
                )
                if self.leading_whitespace_behavior == "gpt2-mixed":
                    search_queries2.append(
                        self.tokenizer.encode_plus(
                            f" {s[idx2[0]:idx2[1]]}", add_special_tokens=False
                        )["input_ids"]
                    )
                elif self.leading_whitespace_behavior == "llama-mixed":
                    search_queries2.append(
                        self.tokenizer.encode_plus(
                            f"{s[idx2[0]:idx2[1]]}", add_special_tokens=False
                        )["input_ids"]
                    )

        # search_queries1 = [self.tokenizer.encode_plus(f'{" ".join(s.split()[idx1[0]:idx1[1]])}', add_special_tokens = False)['input_ids'] for s, idx1, idx2 in sentences]
        # search_queries2 = [self.tokenizer.encode_plus(f'{" ".join(s.split()[idx2[0]:idx2[1]])}', add_special_tokens = False)['input_ids'] for s, idx1, idx2 in sentences]

        query_idx1 = list(
            map(
                lambda x: find_pattern(x[0], x[1], multi_strategy),
                zip(search_queries1, input_ids.tolist()),
            )
        )
        query_idx2 = list(
            map(
                lambda x: find_pattern(x[0], x[1], multi_strategy),
                zip(search_queries2, input_ids.tolist()),
            )
        )

        if multi_strategy in ["first", "last"]:

            query_idx1 = [q[0] for q in query_idx1]
            query_idx2 = [q[0] for q in query_idx2]

            if isinstance(layer, list) or layer == "all":
                # representations = list(map(lambda x: x[torch.arange(num_inputs)[:, None], query_idx].mean(1), hidden_states))
                representations1 = list(
                    map(
                        lambda x: torch.stack(
                            [
                                hs.squeeze()[idx[0] : idx[1]].mean(0)
                                for hs, idx in zip(
                                    x.split([1] * num_inputs), query_idx1
                                )
                            ]
                        ),
                        hidden_states,
                    )
                )
                representations2 = list(
                    map(
                        lambda x: torch.stack(
                            [
                                hs.squeeze()[idx[0] : idx[1]].mean(0)
                                for hs, idx in zip(
                                    x.split([1] * num_inputs), query_idx2
                                )
                            ]
                        ),
                        hidden_states,
                    )
                )
            else:
                if layer is None:
                    layer = self.layers
                elif layer == "static" or "pre":
                    layer = layer
                elif layer > self.layers:
                    raise ValueError(
                        f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!"
                    )
                # representations = hidden_states[torch.arange(num_inputs)[:, None], query_idx].mean(1)
                representations1 = torch.stack(
                    [
                        hs.squeeze()[idx[0] : idx[1]].mean(0)
                        for hs, idx in zip(
                            hidden_states.split([1] * num_inputs), query_idx1
                        )
                    ]
                )
                representations2 = torch.stack(
                    [
                        hs.squeeze()[idx[0] : idx[1]].mean(0)
                        for hs, idx in zip(
                            hidden_states.split([1] * num_inputs), query_idx2
                        )
                    ]
                )
        else:
            if isinstance(layer, list) or layer == "all":
                representations1 = list(
                    map(
                        lambda x: index_states(x, num_inputs, query_idx1), hidden_states
                    )
                )
                representations2 = list(
                    map(
                        lambda x: index_states(x, num_inputs, query_idx2), hidden_states
                    )
                )
            else:
                if layer is None:
                    layer = self.layers
                elif layer == "static" or "pre":
                    layer = layer
                elif layer > self.layers:
                    raise ValueError(
                        f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!"
                    )
                representations1 = index_states(hidden_states, num_inputs, query_idx1)
                representations2 = index_states(hidden_states, num_inputs, query_idx2)

        return representations1, representations2


class EncDecCWE(CWE):
    """
    Implements the contextualized word embedding class to
    facilitate extraction of word representations from an encoder-decoder
    transformer model.
    """

    def __init__(
        self, model_name: str, device: Optional[str] = "cpu", pretrained: bool = True
    ) -> None:
        """
        :param model_name: name of the model, should either be a path
            to a model (.pt or .bin file) stored locally, or a
            pretrained model stored on the Huggingface Model Hub.
        :type model_name: str
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        :param pretrained: whether to load the model with pretrained weights.
            loads a randomly initialized model if `False`. Default = `True`.
        :type pretrained: bool
        """
        super().__init__(model_name, device, pretrained)

    def encode_text(
        self, text: Union[str, List[str]], layer: Union[int, List[int]] = None
    ) -> Tuple:
        """
        Encodes batch of raw sentences using the model to return hidden
        states at a given layer.

        :param ``Union[str, List[str]]`` text: batch of raw sentences
        :param layer: layer from which the hidden states are extracted.
        :type layer: int
        :return: Tuple `(input_ids, hidden_states)`
        """
        sentences = [text] if isinstance(text, str) else text

        # Encode sentence into ids stored in the model's embedding layer(s).
        encoded = self.tokenizer(sentences, padding="longest", return_tensors="pt")
        decoded = self.tokenizer([""] * len(sentences), return_tensors="pt")
        encoded = encoded.to(self.device)
        decoded = decoded.to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        if "cuda" in self.device:
            attention_mask = attention_mask.cpu()

        if layer == "static" or layer == "pre":
            attention_mask = attention_mask.to(self.device)
            hidden_states = [self.model.shared.weight.detach()[i] for i in input_ids]
            hidden_states = torch.stack(hidden_states)
            hidden_states = hidden_states * attention_mask
        else:
            # Compute hidden states for the sentence for the given layer.
            output = self.model(**encoded, decoder_input_ids=decoded.input_ids)

            # Ideally the hidden states would be extracted from the encoder.
            if isinstance(layer, list) or layer == "all":
                hidden_states = output.encoder_hidden_states
                if "cuda" in self.device:
                    input_ids = input_ids.cpu()
                    hidden_states = [h.detach().cpu() for h in hidden_states]
                else:
                    hidden_states = [h.detach() for h in hidden_states]
                if layer != "all":
                    hidden_states = [hidden_states[i] for i in sorted(layer)]

                hidden_states = [h * attention_mask for h in hidden_states]
            else:
                # if layer != 'all':
                if layer is None:
                    layer = self.layers
                elif layer > self.layers:
                    raise ValueError(
                        f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!"
                    )
                hidden_states = output.encoder_hidden_states[layer]
                if "cuda" in self.device:
                    input_ids = input_ids.cpu()
                    hidden_states = hidden_states.detach().cpu()
                else:
                    hidden_states = hidden_states.detach()

                hidden_states = hidden_states * attention_mask

        return input_ids, hidden_states


class VisualCWE(CWE):
    def __init__(
        self, model_name: str, device: Optional[str] = "cpu", **kwargs
    ) -> None:
        """
        :param model_name: name of the model, should either be a path
            to a model (.pt or .bin file) stored locally, or a
            pretrained model stored on the Huggingface Model Hub.
        :type model_name: str
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        :param pretrained: whether to load the model with pretrained weights.
            loads a randomly initialized model if `False`. Default = `True`.
        :type pretrained: bool
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_name, use_fast=True, **kwargs
        )
        self.model_name = model_name
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, output_hidden_states=True, **kwargs
        )

        self.dimensions = list(self.model.parameters())[-1].shape[0]

        self.layers = self.model.config.text_config.num_hidden_layers

        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|pad|>"]}
            )
            self.processor.tokenizer.pad_token = "<|pad|>"
            self.model.resize_token_embeddings(len(self.processor.tokenizer))

        self.leading_whitespace_behavior = leading_whitespace_behavior(
            self.processor.tokenizer
        )

        if self.processor.tokenizer.padding_side == "left":
            self.processor.tokenizer.padding_side = "right"

        self.tokenizer = self.processor.tokenizer

        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, text, images=None, layer=-1):
        sentences = [text] if isinstance(text, str) else text

        if images is not None:
            print("in img mode")
            images = [images] if isinstance(images, Image.Image) else images

            def _chat_template(text):
                context = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text}, {"type": "image"}],
                    }
                ]

                context = self.processor.apply_chat_template(
                    context, continue_final_message=True
                )
                return context

            sentences = [_chat_template(s) for s in sentences]

            encoded = self.processor(
                text=sentences, images=images, return_tensors="pt", padding=True
            )

        else:
            encoded = self.processor(text=sentences, return_tensors="pt", padding=True)

        encoded = encoded.to(self.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        if "cuda" in self.device:
            attention_mask = attention_mask.cpu()

        if layer == "static" or layer == "pre":
            attention_mask = attention_mask.to(self.model.device)
            hidden_states = [
                self.model.get_output_embeddings().weight.detach()[i] for i in input_ids
            ]
            hidden_states = torch.stack(hidden_states)
            hidden_states = hidden_states * attention_mask
        else:
            # Compute hidden states for the sentence for the given layer.
            output = self.model(**encoded)

            # Hidden states appear as the last element of the otherwise custom hidden_states object
            if isinstance(layer, list) or layer == "all":
                hidden_states = output.hidden_states
                if "cuda" in self.device:
                    input_ids = input_ids.cpu()
                    hidden_states = [h.detach().cpu() for h in hidden_states]
                else:
                    hidden_states = [h.detach() for h in hidden_states]
                if layer != "all":
                    hidden_states = [hidden_states[i] for i in sorted(layer)]

                hidden_states = [h * attention_mask for h in hidden_states]
            else:
                if layer is None:
                    layer = self.layers
                elif layer > self.layers:
                    raise ValueError(
                        f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!"
                    )
                hidden_states = output.hidden_states[layer]
                if "cuda" in self.device:
                    input_ids = input_ids.cpu()
                    hidden_states = hidden_states.detach().cpu()
                else:
                    hidden_states = hidden_states.detach()

                hidden_states = hidden_states * attention_mask

        return input_ids, hidden_states

    def extract_representation(
        self,
        sentence_words: Union[
            List[List[Union[str, Union[Tuple[int, int], str]]]],
            List[Union[str, Union[Tuple[int, int], str]]],
        ],
        images=None,
        layer: Union[int, List[int]] = None,
        multi_strategy="last",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Extract representations from the model at a given layer.

        :param ``Union[List[List[Union[str, Union[Tuple(int, int), str]]]], List[Union[str, Union[Tuple(int, int), str]]]]`` sentence_words: Input
            consisting of `[(sentence, word)]`, where sentence is an
            input sentence, and word is a word present in the sentence
            that will be masked out, or `[(sentence, (start, end))]`,
            where (start, end) is a tuple consisting of the character
            span indices that form the word.
        :param ``Union[int, List[int]]`` layer: layer(s) from which the hidden states are extracted.
        :return: torch tensors or list of torch
            tensors corresponding to word representations
        """
        sentences = (
            [sentence_words] if isinstance(sentence_words[0], str) else sentence_words
        )

        num_inputs = len(sentences)

        input_ids, hidden_states = self.encode_text(
            list(list(zip(*sentences))[0]), images, layer
        )

        if isinstance(sentences[0][1], str):
            sentences = [(s, character_span(s, w)) for s, w in sentences]

        search_queries = []
        for s, idx in sentences:
            if 0 in idx:
                search_queries.append(
                    self.processor.tokenizer.encode_plus(
                        f"{s[idx[0]:idx[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif (
                self.leading_whitespace_behavior == "llama"
            ):  ## Seems like the GPT2 approach really fails w Llama's tokenizer
                search_queries.append(
                    self.processor.tokenizer.encode_plus(
                        f"{s[idx[0]:idx[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif self.leading_whitespace_behavior == "gpt2":
                ## this one really matters if we are using GPT2
                search_queries.append(
                    self.processor.tokenizer.encode_plus(
                        f" {s[idx[0]:idx[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif "mixed" in self.leading_whitespace_behavior:
                ## If behavior is mixed, let's stick with the majority of the tokenizer's behavior, but throw in a warning:
                print(
                    "Tokenizer has inconsistent behavior around leading whitespaces: this may lead to issues."
                )
                if self.leading_whitespace_behavior == "gpt2-mixed":
                    search_queries.append(
                        self.processor.tokenizer.encode_plus(
                            f" {s[idx[0]:idx[1]]}", add_special_tokens=False
                        )["input_ids"]
                    )
                elif self.leading_whitespace_behavior == "llama-mixed":
                    search_queries.append(
                        self.processor.tokenizer.encode_plus(
                            f"{s[idx[0]:idx[1]]}", add_special_tokens=False
                        )["input_ids"]
                    )

        query_idx = list(
            map(
                lambda x: find_pattern(x[0], x[1], multi_strategy),
                zip(search_queries, input_ids.tolist()),
            )
        )

        if multi_strategy in ["first", "last"]:

            query_idx = [q[0] for q in query_idx]

            if isinstance(layer, list) or layer == "all":
                representations = list(
                    map(
                        lambda x: torch.stack(
                            [
                                hs.squeeze()[idx[0] : idx[1]].mean(0)
                                for hs, idx in zip(x.split([1] * num_inputs), query_idx)
                            ]
                        ),
                        hidden_states,
                    )
                )
            else:
                if layer is None:
                    layer = self.layers
                elif layer == "static" or "pre":
                    layer = layer
                elif layer > self.layers:
                    raise ValueError(
                        f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!"
                    )
                # representations = hidden_states[torch.arange(num_inputs)[:, None], query_idx].mean(1)
                representations = torch.stack(
                    [
                        hs.squeeze()[idx[0] : idx[1]].mean(0)
                        for hs, idx in zip(
                            hidden_states.split([1] * num_inputs), query_idx
                        )
                    ]
                )
        else:
            if isinstance(layer, list) or layer == "all":
                representations = list(
                    map(lambda x: index_states(x, num_inputs, query_idx), hidden_states)
                )
            else:
                if layer is None:
                    layer = self.layers
                elif layer == "static" or "pre":
                    layer = layer
                elif layer > self.layers:
                    raise ValueError(
                        f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!"
                    )
                representations = index_states(hidden_states, num_inputs, query_idx)

        return representations

    def extract_paired_representations(
        self,
        sentence_words: Union[Tuple[str], List[Tuple[str]]],
        images=None,
        layer: int = None,
        multi_strategy="last",
    ) -> Tuple:
        """
        Extract representations of pairs of words from a given sentence
        from the model at a given layer.

        :param ``Union[Tuple[str], List[Tuple[str]]]`` text: Input
            consisting of `[(sentence, word1, word2)]`, where sentence
            is an input sentence, and word1, word2 are two words
            present in the sentence that will be masked out.
        :param layer: layer from which the representations are
            extracted.
        :type layer: int
        :return: Tuple consisting of torch tensors or lists of torch
            tensors corresponding to word representations
        """
        sentences = (
            [sentence_words] if isinstance(sentence_words[0], str) else sentence_words
        )

        num_inputs = len(sentences)

        input_ids, hidden_states = self.encode_text(
            list(list(zip(*sentences))[0]), images, layer
        )

        if isinstance(sentences[0][1], str):
            sentences = [
                (s, character_span(s, w1), character_span(s, w2))
                for s, w1, w2 in sentences
            ]

        search_queries1 = []
        search_queries2 = []
        for s, idx1, idx2 in sentences:
            if 0 in idx1:
                search_queries1.append(
                    self.tokenizer.encode_plus(
                        f"{s[idx1[0]:idx1[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif (
                self.leading_whitespace_behavior == "llama"
            ):  ## Seems like the GPT2 approach really fails w Llama's tokenizer
                search_queries1.append(
                    self.tokenizer.encode_plus(
                        f"{s[idx1[0]:idx1[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif self.leading_whitespace_behavior == "gpt2":
                ## this one really matters if we are using GPT2
                search_queries1.append(
                    self.tokenizer.encode_plus(
                        f" {s[idx1[0]:idx1[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif "mixed" in self.leading_whitespace_behavior:
                ## If behavior is mixed, let's stick with the majority of the tokenizer's behavior, but throw in a warning:
                print(
                    "Tokenizer has inconsistent behavior around leading whitespaces: this may lead to issues."
                )
                if self.leading_whitespace_behavior == "gpt2-mixed":
                    search_queries1.append(
                        self.tokenizer.encode_plus(
                            f" {s[idx1[0]:idx1[1]]}", add_special_tokens=False
                        )["input_ids"]
                    )
                elif self.leading_whitespace_behavior == "llama-mixed":
                    search_queries1.append(
                        self.tokenizer.encode_plus(
                            f"{s[idx1[0]:idx1[1]]}", add_special_tokens=False
                        )["input_ids"]
                    )
            if 0 in idx2:
                search_queries2.append(
                    self.tokenizer.encode_plus(
                        f"{s[idx2[0]:idx2[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif (
                self.leading_whitespace_behavior == "llama"
            ):  ## Seems like the GPT2 approach really fails w Llama's tokenizer
                search_queries2.append(
                    self.tokenizer.encode_plus(
                        f"{s[idx2[0]:idx2[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif self.leading_whitespace_behavior == "gpt2":
                ## this one really matters if we are using GPT2
                search_queries2.append(
                    self.tokenizer.encode_plus(
                        f" {s[idx2[0]:idx2[1]]}", add_special_tokens=False
                    )["input_ids"]
                )
            elif "mixed" in self.leading_whitespace_behavior:
                ## If behavior is mixed, let's stick with the majority of the tokenizer's behavior, but throw in a warning:
                print(
                    "Tokenizer has inconsistent behavior around leading whitespaces: this may lead to issues."
                )
                if self.leading_whitespace_behavior == "gpt2-mixed":
                    search_queries2.append(
                        self.tokenizer.encode_plus(
                            f" {s[idx2[0]:idx2[1]]}", add_special_tokens=False
                        )["input_ids"]
                    )
                elif self.leading_whitespace_behavior == "llama-mixed":
                    search_queries2.append(
                        self.tokenizer.encode_plus(
                            f"{s[idx2[0]:idx2[1]]}", add_special_tokens=False
                        )["input_ids"]
                    )

        # search_queries1 = [self.tokenizer.encode_plus(f'{" ".join(s.split()[idx1[0]:idx1[1]])}', add_special_tokens = False)['input_ids'] for s, idx1, idx2 in sentences]
        # search_queries2 = [self.tokenizer.encode_plus(f'{" ".join(s.split()[idx2[0]:idx2[1]])}', add_special_tokens = False)['input_ids'] for s, idx1, idx2 in sentences]

        query_idx1 = list(
            map(
                lambda x: find_pattern(x[0], x[1], multi_strategy),
                zip(search_queries1, input_ids.tolist()),
            )
        )
        query_idx2 = list(
            map(
                lambda x: find_pattern(x[0], x[1], multi_strategy),
                zip(search_queries2, input_ids.tolist()),
            )
        )

        if multi_strategy in ["first", "last"]:

            query_idx1 = [q[0] for q in query_idx1]
            query_idx2 = [q[0] for q in query_idx2]

            if isinstance(layer, list) or layer == "all":
                # representations = list(map(lambda x: x[torch.arange(num_inputs)[:, None], query_idx].mean(1), hidden_states))
                representations1 = list(
                    map(
                        lambda x: torch.stack(
                            [
                                hs.squeeze()[idx[0] : idx[1]].mean(0)
                                for hs, idx in zip(
                                    x.split([1] * num_inputs), query_idx1
                                )
                            ]
                        ),
                        hidden_states,
                    )
                )
                representations2 = list(
                    map(
                        lambda x: torch.stack(
                            [
                                hs.squeeze()[idx[0] : idx[1]].mean(0)
                                for hs, idx in zip(
                                    x.split([1] * num_inputs), query_idx2
                                )
                            ]
                        ),
                        hidden_states,
                    )
                )
            else:
                if layer is None:
                    layer = self.layers
                elif layer == "static" or "pre":
                    layer = layer
                elif layer > self.layers:
                    raise ValueError(
                        f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!"
                    )
                # representations = hidden_states[torch.arange(num_inputs)[:, None], query_idx].mean(1)
                representations1 = torch.stack(
                    [
                        hs.squeeze()[idx[0] : idx[1]].mean(0)
                        for hs, idx in zip(
                            hidden_states.split([1] * num_inputs), query_idx1
                        )
                    ]
                )
                representations2 = torch.stack(
                    [
                        hs.squeeze()[idx[0] : idx[1]].mean(0)
                        for hs, idx in zip(
                            hidden_states.split([1] * num_inputs), query_idx2
                        )
                    ]
                )
        else:
            if isinstance(layer, list) or layer == "all":
                representations1 = list(
                    map(
                        lambda x: index_states(x, num_inputs, query_idx1), hidden_states
                    )
                )
                representations2 = list(
                    map(
                        lambda x: index_states(x, num_inputs, query_idx2), hidden_states
                    )
                )
            else:
                if layer is None:
                    layer = self.layers
                elif layer == "static" or "pre":
                    layer = layer
                elif layer > self.layers:
                    raise ValueError(
                        f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!"
                    )
                representations1 = index_states(hidden_states, num_inputs, query_idx1)
                representations2 = index_states(hidden_states, num_inputs, query_idx2)

        return representations1, representations2
