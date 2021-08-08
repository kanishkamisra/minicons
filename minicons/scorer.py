from typing import Iterable, Union, List, Dict, Optional, Callable, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from collections import defaultdict

from itertools import chain
from re import sub

class LMScorer:
    """
    Base LM scorer class intended to store models and tokenizers along
    with methods to facilitate the analysis of language model output scores.
    """
    def __init__(self, model_name: str, device: Optional[str] = 'cpu') -> None:
        """
        :param model_name: name of the model, should either be a path
            to a model (.pt or .bin file) stored locally, or a
            pretrained model stored on the Huggingface Model Hub.
        :type model_name: str
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
        self.device = device
        self.vocab = defaultdict(list)
        {self.vocab[x.strip()].append(i) for x, i in [(self.tokenizer.decode([i]), i) for i in range(self.tokenizer.vocab_size)]}

    def add_special_tokens(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
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

    def query(self, distribution: torch.Tensor, queries: List[str]) -> Tuple:
        # this will be self.vocab tho
        query_ids = [self.vocab[a] for a in queries]
        maxlen = max(map(len, query_ids))
        query_ids = [q + [self.tokenizer.pad_token_id] * (maxlen - len(q)) if len(q) < maxlen else q for q in query_ids]
        current_batch_size = distribution.shape[0]
        probs = distribution[torch.arange(current_batch_size)[:, None], query_ids].max(1).values.exp().tolist()
        
        inv_ranks = distribution.argsort().argsort() + 1
        ranks = distribution.shape[1] - inv_ranks + 1
        token_ranks = ranks[torch.arange(current_batch_size)[:, None], query_ids].min(1).values.tolist()
    
        return probs, token_ranks

    def logprobs(self, batch: Iterable, rank: bool = False) -> Union[float, List[float]]:
        raise NotImplementedError

    def prepare_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        raise NotImplementedError

    def prime_text(self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]]) -> Tuple:
        raise NotImplementedError

    def seq_score(self, batch: Iterable):
        raise NotImplementedError
    
    def score(self, batch: Union[str, List[str]], pool: Callable = torch.mean, *args) -> Union[float, List[float]]:
        '''
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
        '''
        result = self.logprobs(self.prepare_text(batch))
        logprob, _ = list(zip(*result))
        pooled = list(map(lambda x: pool(x, *args).tolist(), logprob))
        
        return pooled
    
    def adapt_score(self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]], pool: Callable = torch.mean, *args) -> Union[float, List[float]]:
        '''
        Pooled estimates of sequence log probabilities, given a
        preamble, computed by the language model. Pooling is usually
        done using a function that is passed to the method.

        :param preamble: a batch of preambles or primes passed to the
            language model. This is what the sequence is conditioned on,
            and the model ignores the word probabilities of this part
            of the input in estimating the overall score.
        :type preamble: Union[str, List[str]]
        :param stimuli: a batch of sequences (same length as preamble)
            that form the main input consisting of the sequence whose
            score you want to calculate.
        :type stimuli: Union[str, List[str]]
        :param pool: Pooling function, is selected to be
            `torch.mean()` by default.
        :type pool: Callable
        :return: Float or list of floats specifying the log
            probabilities of the input sentence(s). 
        :rtype: Union[float, List[float]]
        '''
        result = self.logprobs(self.prime_text(preamble, stimuli))
        logprob, _ = list(zip(*result))
        poold = list(map(lambda x: pool(x, *args).tolist(), logprob))
        
        return poold

    def encode(self, text: Union[str, List[str]], manual_special: bool = True, return_tensors: Optional[str] = 'pt') -> Dict:
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
        """
        sentences = [text] if isinstance(text, str) else text

        if manual_special:
            # manually add special tokens
            sentences = self.add_special_tokens(sentences)
            if return_tensors:
                tokens = self.tokenizer.batch_encode_plus(sentences, add_special_tokens = False, padding = 'longest', return_attention_mask = True, return_tensors = return_tensors)
        else:
            # mostly for masked LMs
            tokens = self.tokenizer.batch_encode_plus(sentences, padding = 'longest', return_attention_mask = True)

        return tokens
    
    def decode(self, idx):
        """
        Decode input ids using the model's tokenizer.

        :param idx: list of ids.
        :return: Encoded batch 
        """
        return [self.tokenizer.decode([x]).strip() for x in self.tokenizer.convert_tokens_to_ids(self.tokenizer.convert_ids_to_tokens(idx))]

class MaskedLMScorer(LMScorer):
    """
    Implements LM scoring and output probability analysis for masked
    language models such as BERT and RoBERTa.
    """
    def __init__(self, model_name: str, device: Optional[str] = 'cpu') -> None:
        """
        :param model_name: name of the model, should either be a path
            to a model (.pt or .bin file) stored locally, or a
            pretrained model stored on the Huggingface Model Hub.

        :type model_name: str
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        """
        super(MaskedLMScorer, self).__init__(model_name, device)
        
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, return_dict = True)
        self.model.to(self.device)
        self.model.eval()
        
        # define CLS and SEP tokens
        self.bos_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
    
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
        sentences = [self.tokenizer.cls_token + " " + sentence + " " + self.tokenizer.sep_token for sentence in sentences]

        return sentences

    def mask(self, sentence_words: Union[Tuple[str], List[Tuple[str]]]) -> Tuple:
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
        sentence_words = [sentence_words] if isinstance(sentence_words[0], str) else sentence_words
        sentences, words = list(zip(*sentence_words))
        words = list(words)
        length = len(words)

        sentences = [sub(rf'(?<![\w\/-])({word})(?=[^\w\/-])', self.tokenizer.mask_token, sentence) for sentence, word in sentence_words]

        return (sentences, words, length)

    def cloze(self, sentence_words: Union[str, List[str]]) -> torch.Tensor:
        """
        Runs inference on masked input. 
        Note: only works for masked LMs.

        :param ``Union[Tuple[str], List[Tuple[str]]]`` sentence_words:
            Input consisting of `[(sentence, word)]`, where sentence
            is an input sentence, and word is a word present in the
            sentence that will be masked out and inferred.
        
        :return: A tensor with log probabilities for the desired word
            in context
        """
        sentences, words, length = self.mask(sentence_words)

        encoded = self.tokenizer(sentences, return_tensors='pt')
        encoded = encoded.to(self.device)

        idx = torch.nonzero(encoded['input_ids'] == self.tokenizer.mask_token_id, as_tuple=False)[:,1].unsqueeze(1)
        word_idx = self.tokenizer(words, add_special_tokens=False)['input_ids']
        with torch.no_grad():
            masked_logits = self.model(**encoded).logits[torch.arange(length)[:, None], idx].squeeze().detach()
            if len(sentences) > 1:
                logprobs = masked_logits - masked_logits.logsumexp(1).unsqueeze(1)
                masked_logprobs = logprobs[torch.arange(len(sentences))[:, None], word_idx].exp().squeeze()
            else:
                logprobs = masked_logits - masked_logits.logsumexp(0)
                masked_logprobs = logprobs[word_idx].exp().squeeze()

        return masked_logprobs


    def prepare_text(self, text: Union[str, List[str]]) -> Iterable[Any]:
        """
        Prepares a batch of input text into a format fit to run MLM
        scoring on. 

        Borrows preprocessing algorithm from Salazar et al. (2020), and
        modifies code from the following github repository by simonpri:
            https://github.com/simonepri/lm-scorer
        
        :param text: batch of sentences to be prepared for scoring.
        :return: Batch of formatted input that can be passed to
            `logprob`
        """
        # converts input text to batch of tensors with every position except the cls and sep token masked
        sentences = [text] if isinstance(text, str) else text
        
        # idea is to tokenize and then create batches of tokenized instances,
        # but with each token in the sequence replaced by the mask token. 
        
        encoded = self.encode(sentences, manual_special = False)

        token_idx = encoded['input_ids']
        attention_masks = encoded['attention_mask']

        masked_tensors = [] # token ids, attention masks, lengths

        for token_ids, attention_mask in zip(token_idx, attention_masks):
            token_ids = torch.tensor(token_ids)
            # final_lengths = len(token_ids) - 2
            attention_mask = torch.tensor(attention_mask)
            
            token_ids_masked_list = []
            attention_masked_list = []

            effective_token_ids = [token for token in token_ids if token != self.pad_token_id and token != self.cls_token_id and token != self.sep_token_id]
            effective_length = len(effective_token_ids)
            

            mask_indices = []
            mask_indices = [[mask_pos] for mask_pos in range(effective_length+2)]

            # We don't mask the [CLS], [SEP] for now for PLL
            mask_indices = mask_indices[1:-1]

            mask_token_id = self.mask_token_id
            for mask_set in mask_indices:
                token_ids_masked = token_ids.clone()
                token_ids_masked[mask_set] = mask_token_id
                attention_masked = attention_mask.clone()
                
                attention_masked_list.append(attention_masked)
                token_ids_masked_list.append(token_ids_masked)
            masked_tensors.append((torch.stack(token_ids_masked_list), torch.stack(attention_masked_list), effective_token_ids, len(mask_indices), 1))
        
        return masked_tensors

    def prime_text(self, preamble: Union[str, List[str]] , stimuli: Union[str, List[str]]) -> Iterable[Any]:
        """
        Prepares a batch of input text `(preamble, stimuli)` into a
        format fit for running MLM scoring on.
        This is different from `prepare_text()` as it runs a different
        preprocessing step, to ensure the preamble is only used as a
        condition for estimating stimuli word probabilities.

        Borrows preprocessing algorithm from Salazar et al. (2020), and
        modifies code from the following github repository by simonpri:
            https://github.com/simonepri/lm-scorer
        
        :param text: batch of sentences to be prepared for scoring.
        :return: Batch of formatted input that can be passed to
            `logprob`
        """
        preamble_text = [preamble] if isinstance(preamble, str) else preamble
        preamble_encoded = self.encode(preamble_text, False)['input_ids']
        preamble_lens = []
        for preamble_tokens in preamble_encoded:
            preamble_lens.append(len([token for token in preamble_tokens if token != self.pad_token_id and token != self.sep_token_id]))
        
        sentences = [preamble + " " + stimuli] if isinstance(preamble, str) else [p + " " + s for p, s in list(zip(preamble, stimuli))]
            
        # idea is to tokenize and then create batches of tokenized instances,
        # but with each token in the sequence replaced by the mask token. 

        encoded = self.encode(sentences, manual_special = False)

        token_idx = encoded['input_ids']
        attention_masks = encoded['attention_mask']

        masked_tensors = [] # token ids, attention masks, lengths

        for i, (token_ids, attention_mask) in enumerate(zip(token_idx, attention_masks)):
            token_ids = torch.tensor(token_ids)
            # final_lengths = len(token_ids) - 2
            attention_mask = torch.tensor(attention_mask)

            token_ids_masked_list = []
            attention_masked_list = []
            
            effective_token_ids = [token for j, token in enumerate(token_ids) if token != self.pad_token_id and token != self.cls_token_id and token != self.sep_token_id and j >= preamble_lens[i]]
            effective_length = len(effective_token_ids) + preamble_lens[i]


            mask_indices = []
            mask_indices = [[mask_pos] for mask_pos in range(preamble_lens[i], effective_length+1)]

            # We don't mask the [CLS], [SEP] for now for PLL
            mask_indices = mask_indices[:-1]

            mask_token_id = self.mask_token_id
            for mask_set in mask_indices:
                token_ids_masked = token_ids.clone()
                token_ids_masked[mask_set] = mask_token_id
                attention_masked = attention_mask.clone()

                attention_masked_list.append(attention_masked)
                token_ids_masked_list.append(token_ids_masked)
            masked_tensors.append((torch.stack(token_ids_masked_list), torch.stack(attention_masked_list), effective_token_ids, len(mask_indices), preamble_lens[i]))

        return masked_tensors

    def distribution(self, batch: Iterable) -> torch.Tensor:
        """
        Returns a distribution over the vocabulary of the model.

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.

        :return: Tensor consisting of log probabilies over vocab items.
        """
        # takes in prepared text and returns scores for each sentence in batch
        token_ids, attention_masks, effective_token_ids, lengths, offsets = list(zip(*batch))
        token_ids = torch.cat(token_ids)
        attention_masks = torch.cat(attention_masks)
        token_ids = token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        effective_token_ids = torch.cat([torch.tensor(x) for x in effective_token_ids])

        sent_tokens = list(map(lambda x: self.tokenizer.convert_ids_to_tokens(x.tolist()), effective_token_ids.split(lengths)))

        indices = list(chain.from_iterable([list(range(o,o+n)) for n, o in zip(lengths, offsets)]))
        with torch.no_grad():
            output = self.model(token_ids, attention_mask = attention_masks)
            logits = output.logits[torch.arange(sum(lengths)), indices]
            if self.device == 'cuda:0' or self.device == "cuda:1":
                logits.detach()

        return logits

    def cloze_distribution(self, queries: Iterable) -> torch.Tensor:
    
        '''
        Accepts as input batch of [(s_i, bw_i)] where s_i is a prompt with an
        abstract token (bw_i) representing a blank word and returns a distribution
        over the vocabulary of the model.

        :param `Iterable` queries: A batch of [(s_i, bw_i)] where s_i is a prompt
            with an abstract token (bw_i) representing a blank word

        :return: Tensor contisting of log probabilities over vocab items.
        '''
        
        queries = [queries] if isinstance(queries[0], str) else queries
        prompts, words = list(zip(*queries))
            
        modified_prompts = self.add_special_tokens(prompts)
        splits = [prompt.split(word) for prompt, word in zip(modified_prompts, words)]
        splits = [[x.strip() for x in s] for s in splits]
        pre, post = list(zip(*splits))
        pre_idx = self.tokenizer(list(pre), add_special_tokens = False, padding=False)['input_ids']
        mask_idx = [len(item) for item in pre_idx]
        masked = [m.replace(w, self.tokenizer.mask_token) for m, w in zip(modified_prompts, words)]
        
        with torch.no_grad():
            encoded = self.tokenizer(masked, add_special_tokens = False, return_tensors='pt', padding = True)
            encoded = encoded.to(self.device)
            logits = self.model(**encoded)
            presoftmax = logits.logits[torch.arange(len(queries)), mask_idx]
            if 'cuda' in self.device:
                presoftmax = presoftmax.detach().cpu()
            else:
                presoftmax = presoftmax.detach()
            
        logprobs = presoftmax - presoftmax.logsumexp(1).unsqueeze(1)
        
        return logprobs    

    def logprobs(self, batch: Iterable, rank = False) -> Union[List[Tuple[torch.Tensor, str]], List[Tuple[torch.Tensor, str, int]]]:
        """
        Returns log probabilities

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.
        :param rank: Specifies whether to also return ranks of words.
        :type rank: bool

        :return: List of MLM score metrics and tokens.
        :rtype: Union[List[Tuple[torch.Tensor, str]], List[Tuple[torch.Tensor, str, int]]]
        """
        token_ids, attention_masks, effective_token_ids, lengths, offsets = list(zip(*batch))
        token_ids = torch.cat(token_ids)
        attention_masks = torch.cat(attention_masks)
        token_ids = token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        effective_token_ids = torch.cat([torch.tensor(x) for x in effective_token_ids])

        sent_tokens = list(map(lambda x: self.tokenizer.convert_ids_to_tokens(x.tolist()), effective_token_ids.split(lengths)))
        
        indices = list(chain.from_iterable([list(range(o,o+n)) for n, o in zip(lengths, offsets)]))
        with torch.no_grad():
            output = self.model(token_ids, attention_mask = attention_masks)
            logits = output.logits[torch.arange(sum(lengths)), indices]
            if self.device == 'cuda:0' or self.device == "cuda:1":
                logits.detach()
            
            sent_log_probs = logits - logits.logsumexp(1).unsqueeze(1)
            if rank:
                shape = sent_log_probs.shape
                inv_ranks = (sent_log_probs).argsort().argsort() + 1
                ranks = shape[1] - inv_ranks + 1
                word_ranks = ranks[torch.arange(shape[0]), effective_token_ids].split(lengths)
            sent_log_probs = sent_log_probs[torch.arange(sum(lengths)), effective_token_ids].type(torch.DoubleTensor).split(lengths)
            # sentence_scores = list(map(lambda x: x.sum().tolist(), logprobs))
            # outputs.append((logprobs, sent_tokens))
            if rank:
                return list(zip(sent_log_probs, sent_tokens, word_ranks))
    
            
        return list(zip(sent_log_probs, sent_tokens))

class IncrementalLMScorer(LMScorer):
    """
    Implements LM scoring and output probability analysis for incremental
    LMs such as GPT and GPT2.
    """
    def __init__(self, model_name: str, device: Optional[str] = 'cpu') -> None:
        """
        :param model_name: name of the model, should either be a path
            to a model (.pt or .bin file) stored locally, or a
            pretrained model stored on the Huggingface Model Hub.

        :type model_name: str
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        """
        super(IncrementalLMScorer, self).__init__(model_name, device)
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, return_dict = True)
        
        # define CLS and SEP tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
            self.tokenizer.pad_token = "<|pad|>"

        # if self.tokenizer.eos_token is None:
        #     self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|eos|>"]})
        #     self.tokenizer.eos_token = "<|eos|>"

        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|bos|>"]})
            self.tokenizer.bos_token = "<|bos|>"

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()
    
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
    
    def prepare_text(self, text: Union[str, List[str]]) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on. 

        :param text: batch of sentences to be prepared for scoring.
        :return: Batch of formatted input that can be passed to
            `logprob`
        """
        encoded = self.encode(text)
        offsets = [0] * len(encoded['input_ids'])
        return encoded, offsets
    
    def prime_text(self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]]) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on. 

        :param text: batch of sentences to be prepared for scoring.
        :return: Batch of formatted input that can be passed to
            `logprob`
        """
        preamble_text = [preamble] if isinstance(preamble, str) else preamble
        preamble_encoded = self.encode(preamble_text, False)['input_ids']
        preamble_lens = []
        for preamble_tokens in preamble_encoded:
            preamble_lens.append(len([token for token in preamble_tokens if token != self.tokenizer.pad_token_id and token != self.tokenizer.sep_token_id]))
        
        sentences = [preamble + " " + stimuli] if isinstance(preamble, str) else [p + " " + s for p , s in list(zip(preamble, stimuli))]
            
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
        ids = ids.to(self.device)
        attention_masks = batch["attention_mask"]
        attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_masks)
            logits = outputs.logits
            if self.device == 'cuda:0' or self.device == "cuda:1":
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

            outputs.append(sent_logits[-1])
        return torch.stack(outputs, 0)

    def next_word_distribution(self, queries: List):
        encoded = self.prime_text(queries, ['the'] * len(queries))
        logits = self.distribution(encoded)
        logprobs = logits - logits.logsumexp(1).unsqueeze(1)
        
        return logprobs

    def logprobs(self, batch: Iterable, rank = False) -> Union[float, List[float]]:
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
        batch, offsets = batch
        ids = batch["input_ids"]
        ids = ids.to(self.device)
        attention_masks = batch["attention_mask"]
        attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_masks)
            logits = outputs.logits
            if self.device == 'cuda:0' or self.device == "cuda:1":
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
            
            sent_log_probs = sent_log_probs.type(torch.DoubleTensor)
            sent_log_probs = sent_log_probs[offsets[sent_index]:]
            lengths = len(sent_log_probs)
            if rank:
                shape = sent_logits.shape
                inv_ranks = (sent_logits).argsort().argsort() + 1
                ranks = shape[1] - inv_ranks + 1
                word_ranks = ranks[list(range(shape[0]))[offsets[sent_index]:], sent_ids[offsets[sent_index]: ].tolist()].split(lengths)
                word_ranks = [x[0] for x in word_ranks]
                outputs.append((sent_log_probs, sent_tokens, word_ranks))
            else:
                outputs.append((sent_log_probs, sent_tokens))
            # output = (sent_log_probs.sum(), sent_ids, sent_tokens)
            # outputs.append(output)
        return outputs