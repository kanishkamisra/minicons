from typing import Iterable, Union, List, Dict, Optional, Callable, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

import itertools

class LMScorer:
    def __init__(self, model_name: str, device: str = 'cpu') -> None:
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
        self.device = device

    def add_special_tokens(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        sentences = [text] if isinstance(text, str) else text
        sentences = [self.tokenizer.bos_token + sentence + self.tokenizer.eos_token for sentence in sentences]

        return sentences

    def logprobs(self, batch: Iterable) -> Union[float, List[float]]:
        raise NotImplementedError

    def prepare_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        raise NotImplementedError

    def prime_text(self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]]) -> Tuple:
        raise NotImplementedError

    def seq_score(self, batch: Iterable):
        raise NotImplementedError
    
    def score(self, batch: Iterable, reduce: Callable = torch.mean, *args) -> Union[float, List[float]]:
        
        result = self.logprobs(self.prepare_text(batch))
        logprob, _ = list(zip(*result))
        reduced = list(map(lambda x: reduce(x, *args).tolist(), logprob))
        
        return reduced
    
    def adapt_score(self, preamble: Iterable, stimuli: Iterable, reduce: Callable = torch.mean, *args) -> Union[float, List[float]]:
        
        result = self.logprobs(self.prime_text(preamble, stimuli))
        logprob, _ = list(zip(*result))
        reduced = list(map(lambda x: reduce(x, *args).tolist(), logprob))
        
        return reduced

    def encode(self, text: Union[str, List[str]], manual_special: bool = True, return_tensors: Optional[str] = 'pt') -> Dict:       
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

class MaskedLMScorer(LMScorer):
    def __init__(self, model_name: str, device: str) -> None:
        
        super(MaskedLMScorer, self).__init__(model_name, device)
        
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, return_dict = True)
        self.model.to(self.device)
        self.model.eval()
        
        # define CLS and SEP tokens
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def prepare_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
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
            masked_tensors.append((torch.stack(token_ids_masked_list), torch.stack(attention_masked_list), effective_token_ids, len(mask_indices), 0))
        
        return masked_tensors

    def prime_text(self, preamble: Union[str, List[str]] , stimuli: Union[str, List[str]]) -> Tuple:
   
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

    def logprobs(self, batch: Iterable, rank = False) -> Union[float, List[float]]:
        
        # takes in prepared text and returns scores for each sentence in batch
        token_ids, attention_masks, effective_token_ids, lengths, offsets = list(zip(*batch))
        token_ids = torch.cat(token_ids)
        attention_masks = torch.cat(attention_masks)
        token_ids = token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        effective_token_ids = torch.cat([torch.tensor(x) for x in effective_token_ids])

        sent_tokens = list(map(lambda x: self.tokenizer.convert_ids_to_tokens(x.tolist()), effective_token_ids.split(lengths)))
        
        indices = list(itertools.chain.from_iterable([list(range(o,o+n)) for n, o in zip(lengths, offsets)]))
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
    def __init__(self, model_name: str, device: str) -> None:
        
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
        sentences = [text] if isinstance(text, str) else text
        sentences = [self.tokenizer.bos_token + sentence for sentence in sentences]

        return sentences
    
    def prepare_text(self, text: Union[str, List[str]]) -> Tuple:
        encoded = self.encode(text)
        offsets = [0] * len(encoded['input_ids'])
        return encoded, offsets
    
    def prime_text(self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]]) -> Tuple:
        preamble_text = [preamble] if isinstance(preamble, str) else preamble
        preamble_encoded = self.encode(preamble_text, False)['input_ids']
        preamble_lens = []
        for preamble_tokens in preamble_encoded:
            preamble_lens.append(len([token for token in preamble_tokens if token != self.tokenizer.pad_token_id and token != self.tokenizer.sep_token_id]))
        
        sentences = [preamble + " " + stimuli] if isinstance(preamble, str) else [p + " " + s for p , s in list(zip(preamble, stimuli))]
            
        return self.encode(sentences), preamble_lens

    def logprobs(self, batch: Iterable, rank = False) -> Union[float, List[float]]:
        
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
                outputs.append((sent_log_probs, sent_tokens, word_ranks))
            else:
                outputs.append((sent_log_probs, sent_tokens))
            # output = (sent_log_probs.sum(), sent_ids, sent_tokens)
            # outputs.append(output)
        return outputs