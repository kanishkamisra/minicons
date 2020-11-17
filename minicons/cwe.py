from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch

class CWE():
    def __init__(self, model: str, all_layers: bool = False, device: str = "cpu") -> None:
        
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        # self.vocab = self.tokenizer.vocab

        self.config = AutoConfig.from_pretrained(self.model, output_hidden_states = all_layers)

        self.layers = self.config.num_hidden_layers

        self.transformer = AutoModel.from_config(self.config)
        self.transformer.to(device)
        self.transformer.eval()

    def tokenize(self, sentence: str) -> list:
        return self.tokenizer.tokenize(sentence)

    # def encode(self, sentence: list):
    #     word2ids = self.tokenizer.batch_encode_plus(sentence, pad_to_max_length = True, return_tensors = "pt")
    #     # if "gpt" in 
    #     return word2ids
    
    def encode_sentence(self, sentence: str, layer: int = None):

        if layer is None:
            layer = self.layers
        # Encode sentence into ids stored in the model's embedding layer(s).
        encoded = self.tokenizer.batch_encode_plus([sentence], return_tensors="pt")
        input_ids = encoded["input_ids"]

        # Compute hidden states for the sentence for the given layer.
        hidden_states = self.transformer(input_ids = input_ids)

        # Hidden states appear as the last element of the otherwise custom hidden_states object
        hidden_states = hidden_states[-1][layer]

        return encoded, hidden_states

    def context_cosine(self, sentence: str, word: str, layer: int = None):

        if layer is None:
            layer = self.layers

        encoded, hidden_states = self.encode_sentence(sentence, layer)
        input_ids = encoded['input_ids']

        word_id = self.tokenizer.encode(word, add_special_tokens = False, add_prefix_space = True)

        if len(word_id) > 1:
            raise ValueError("Word not in Transformer's Vocabulary")
        else:
            word_id = word_id[0]

        word_idx = (input_ids == word_id).nonzero().flatten()[1]

        sentence_idx = list(range(0, input_ids.shape[1]))

        # only select ids that do not correspond to the word.
        context_idx = sentence_idx[0:word_idx] + sentence_idx[word_idx + 1:]
        context_ids = input_ids[0].tolist()[0:word_idx] + input_ids[0].tolist()[word_idx + 1:]

        context = hidden_states[0, context_idx, :]
        word_rep = hidden_states[:, word_idx, :]

        cosines = torch.nn.functional.cosine_similarity(word_rep, context).tolist()

        tokens = [self.tokenizer.convert_ids_to_tokens(x) for x in context_ids]
        words = ["".join(self.tokenizer.convert_tokens_to_string(x).split()) for x in tokens]

        return words, cosines
