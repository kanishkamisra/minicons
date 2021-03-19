from typing import Iterable, Union, List, Dict, Optional, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class SupervisedHead:
    """
    (Under construction)
    Implements the supervised head class to facilitate behavioral
    analyses of model outputs.
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
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict = True,)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, inputs: Union[str, List[str]], return_tensors: Optional[str] = 'pt') -> Dict:
        """
        Encodes batch of inputs to the supervised model to return
        an encoded format to be passed to the model.

        :param ``Union[str, List[str]]`` inputs: batch of inputs to be
            encoded by the model.
        :param ``Optional[str]`` return_tensors: returned tensor format.
            Default `'pt'`
        :return: Dictionary of encoded input.
        """
        inputs = [inputs] if isinstance(inputs, str) else inputs

        encoded = self.tokenizer(inputs, padding = 'longest', return_tensors = 'pt')
        if self.device != 'cpu':
            encoded = encoded.to(self.device)
        
        return encoded
    
    def logits(self, inputs: Union[str, List[str]], probs: bool = True, verbose: bool = False) -> Union[torch.Tensor, Dict]:
        """
        Runs inference on the model and returns logits for each label
        depending on the supervised task on which the model was trained
        on.

        :param ``Union[str, List[str]]`` inputs: batch of inputs to be
            encoded by the model.
        :param probs: specifies whether to return probabilities.
        :type probs: bool
        :param verbose: specifies if the label names should be revealed
            in the output (if they exist).
        :type  verbose: bool
        :return: Either a torch tensor consisting of the model outputs
            or a dictionary consisting of `{label: probability}`.
        """
        encoded = self.encode(inputs)

        output = self.model(**encoded).logits.detach()

        if probs:
            output = output.softmax(1)

        if verbose:
            if 'label2id' not in self.model.config.to_dict().keys():
                output = output
            else:
                output = {k: output[:, v].tolist()for k, v in self.model.config.to_dict()['label2id'].items()}

        return output
        
