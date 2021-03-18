from typing import Iterable, Union, List, Dict, Optional, Tuple

from .utils import find_pattern, find_index, find_paired_indices

from transformers import AutoModel, AutoTokenizer
import torch

class CWE():
    def __init__(self, model_name: str, device: str = 'cpu') -> None:

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True, add_prefix_space = True)
        self.model = AutoModel.from_pretrained(model_name, return_dict = True, output_hidden_states = True)

        self.layers = self.model.config.num_hidden_layers

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
            self.tokenizer.pad_token = "<|pad|>"
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, text: Union[str, List[str]], layer: int = None) -> Tuple:

        sentences = [text] if isinstance(text, str) else text

        # Encode sentence into ids stored in the model's embedding layer(s).
        encoded = self.tokenizer.batch_encode_plus(sentences, padding = 'longest', return_tensors="pt")
        encoded = encoded.to(self.device)

        input_ids = encoded['input_ids']

        # Compute hidden states for the sentence for the given layer.
        output = self.model(**encoded)

#         # Hidden states appear as the last element of the otherwise custom hidden_states object
        if layer != 'all':
            if layer is None:
                layer = self.layers
            elif layer > self.layers:
                raise ValueError(f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!")
            hidden_states = output.hidden_states[layer]
            if "cuda" in self.device:
                input_ids = input_ids.cpu()
                hidden_states = hidden_states.detach().cpu()
            else:
                hidden_states = hidden_states.detach()
        else:
            hidden_states = output.hidden_states
        
            if "cuda" in self.device:
                input_ids = input_ids.cpu()
                hidden_states = [h.detach().cpu() for h in hidden_states]
            else:
                hidden_states = [h.detach() for h in hidden_states]

        return input_ids, hidden_states
    
    # A function that extracts the representation of a given word in a sentence (first occurrence only)

    def extract_representation(self, text, layer:int = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        
        sentences = [text] if isinstance(text[0], str) else text

        num_inputs = len(sentences)

        input_ids, hidden_states = self.encode_text(list(list(zip(*sentences))[0]), layer)

        if isinstance(sentences[0][1], str):
            sentences = [(s, find_index(s, w)) for s, w in sentences]
    
        search_queries = [self.tokenizer.encode_plus(f'{" ".join(s.split()[idx[0]:idx[1]])}', add_special_tokens = False)['input_ids'] for s, idx in sentences]

        query_idx = list(map(lambda x: find_pattern(x[0], x[1]), zip(search_queries, input_ids.tolist())))

        if layer != 'all':
            if layer is None:
                layer = self.layers
            elif layer > self.layers:
                raise ValueError(f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!")
            representations = hidden_states[torch.arange(num_inputs)[:, None], query_idx].mean(1)
        else:
            representations = list(map(lambda x: x[torch.arange(num_inputs)[:, None], query_idx].mean(1), hidden_states))
        
        return representations

    def extract_paired_representations(self, text, layer:int = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        '''
            Given input of the form (<sentence>, <word1>, <word2>) produce paired tensors that represent the words in the sentence, as encoded by the model
        '''
        sentences = [text] if isinstance(text[0], str) else text

        num_inputs = len(sentences)

        input_ids, hidden_states = self.encode_text(list(list(zip(*sentences))[0]), layer)

        if isinstance(sentences[0][1], str):
            sentences = [(s, *find_paired_indices(s, w1, w2)) for s, w1, w2 in sentences]
        
        search_queries1 = [self.tokenizer.encode_plus(f'{" ".join(s.split()[idx1[0]:idx1[1]])}', add_special_tokens = False)['input_ids'] for s, idx1, idx2 in sentences]
        search_queries2 = [self.tokenizer.encode_plus(f'{" ".join(s.split()[idx2[0]:idx2[1]])}', add_special_tokens = False)['input_ids'] for s, idx1, idx2 in sentences]

        query_idx1 = list(map(lambda x: find_pattern(x[0], x[1]), zip(search_queries1, input_ids.tolist())))
        query_idx2 = list(map(lambda x: find_pattern(x[0], x[1]), zip(search_queries2, input_ids.tolist())))

        if layer != 'all':
            if layer is None:
                layer = self.layers
            elif layer > self.layers:
                raise ValueError(f"Number of layers specified ({layer}) exceed layers in model ({self.layers})!")
            representations1 = hidden_states[torch.arange(num_inputs)[:, None], query_idx1].mean(1)
            representations2 = hidden_states[torch.arange(num_inputs)[:, None], query_idx2].mean(1)
        else:
            representations1 = list(map(lambda x: x[torch.arange(num_inputs)[:, None], query_idx1].mean(1), hidden_states))
            representations2 = list(map(lambda x: x[torch.arange(num_inputs)[:, None], query_idx2].mean(1), hidden_states))
        
        return representations1, representations2