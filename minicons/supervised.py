from typing import Iterable, Union, List, Dict, Optional, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class SupervisedHead:
    def __init__(self, model_name: str, device: str = 'cpu') -> None:
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict = True,)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, inputs: Union[str, List[str]], return_tensors: Optional[str] = 'pt') -> Dict:
        inputs = [inputs] if isinstance(inputs, str) else inputs

        encoded = self.tokenizer(inputs, padding = 'longest', return_tensors = 'pt')
        if self.device != 'cpu':
            encoded = encoded.to(self.device)
        
        return encoded
    
    def logits(self, inputs: Union[str, List[str]], verbose: bool = False) -> Union[torch.Tensor, Dict]:
        encoded = self.encode(inputs)

        output = self.model(**encoded).logits.detach()

        if verbose:
            if 'label2id' not in self.model.config.to_dict().keys():
                output = output
            else:
                output = {k: output[:, v] for k, v in self.model.config.to_dict().items()}

        return output
        
