# minicons: Enabling Flexible Behavioral and Representational Analyses of Transformer Language Models

[![Downloads](https://static.pepy.tech/personalized-badge/minicons?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/minicons)

This repo is a wrapper around the `transformers` [library](https://huggingface.co/transformers) from Hugging Face :hugs:

<!-- TODO: Description-->



## Installation

Install from Pypi using:

```pip install minicons```

## Supported Functionality

- Extract word representations from Contextualized Word Embeddings
- Score sequences using language model scoring techniques, including masked language models following [Salazar et al. (2020)](https://www.aclweb.org/anthology/2020.acl-main.240.pdf), and state space models (such as Mamba).
- Score sequences using VLM models (see below)
- Do scoring in a quantized, multi-gpu setting.


## Examples

1. Extract word representations from contextualized word embeddings:

```py
from minicons import cwe

model = cwe.CWE('bert-base-uncased')

context_words = [("I went to the bank to withdraw money.", "bank"), 
                 ("i was at the bank of the river ganga!", "bank")]

print(model.extract_representation(context_words, layer = 12))

''' 
tensor([[ 0.5399, -0.2461, -0.0968,  ..., -0.4670, -0.5312, -0.0549],
        [-0.8258, -0.4308,  0.2744,  ..., -0.5987, -0.6984,  0.2087]],
       grad_fn=<MeanBackward1>)
'''

# if model is seq2seq:
model = cwe.EncDecCWE('t5-small')

print(model.extract_representation(context_words))

'''(last layer, by default)
tensor([[-0.0895,  0.0758,  0.0753,  ...,  0.0130, -0.1093, -0.2354],
        [-0.0695,  0.1142,  0.0803,  ...,  0.0807, -0.1139, -0.2888]])
'''
```

2. Compute sentence acceptability measures (surprisals) using Language Models:

```py
from minicons import scorer

mlm_model = scorer.MaskedLMScorer('bert-base-uncased', 'cpu')
ilm_model = scorer.IncrementalLMScorer('distilgpt2', 'cpu')

stimuli = ["The keys to the cabinet are on the table.",
           "The keys to the cabinet is on the table."]

# use sequence_score with different reduction options: 
# Sequence Surprisal - lambda x: -x.sum(0).item()
# Sequence Log-probability - lambda x: x.sum(0).item()
# Sequence Surprisal, normalized by number of tokens - lambda x: -x.mean(0).item()
# Sequence Log-probability, normalized by number of tokens - lambda x: x.mean(0).item()
# and so on...

print(ilm_model.sequence_score(stimuli, reduction = lambda x: -x.sum(0).item()))

'''
[39.879737854003906, 42.75846481323242]
'''

# MLM scoring, inspired by Salazar et al., 2020
print(mlm_model.sequence_score(stimuli, reduction = lambda x: -x.sum(0).item()))
'''
[13.962685585021973, 23.415111541748047]
'''
```

3. Computing conditional sequence scoring using LMs


```py
s2s_model = scorer.Seq2SeqScorer('t5-base', 'cpu')

# sequence scoring for batch of input, output, by default = logprobs, can change to other quantities as needed (see minicons readme)
s2s_model.conditional_score(["What is the capital of France?", "What is the capital of France?"], ["Paris.", "Lyon."]) # the same thing works with ilm_model and mlm_model as well

'''OUTPUT:
[-6.089522838592529, -8.20227336883545]
''' 

# Token-wise score of the output queries: -- <pad> token is given a score of 0.0, pass rank=True to also give token ranks
s2s_model.conditional_token_score(["What is the capital of France?", "What is the capital of France?"], ["Paris.", "Lyon."], rank=True) 

'''OUTPUT:
[[('<pad>', 0.0, 0),
  ('Paris', -7.5618486404418945, 168),
  ('.', -4.617197036743164, 11)],
 [('<pad>', 0.0, 0),
  ('Lyon', -12.044157981872559, 3459),
  ('.', -4.36038875579834, 8)]]
'''
```

## A better version of MLM Scoring by Kauf and Ivanova

This version leverages a locally-autoregressive scoring strategy to avoid the overestimation of probabilities of tokens in multi-token words (e.g., "ostrich" -> "ostr" + "#ich"). In particular, tokens probabilities are estimated using the bidirectional context, excluding any future tokens that belong to the same word as the current target token.

For more details, refer to [Kauf and Ivanova, 2023](https://arxiv.org/abs/2305.10588)

```py
from minicons import scorer
mlm_model = scorer.MaskedLMScorer('bert-base-uncased', 'cpu')

stimuli = ['The traveler lost the souvenir.']

# un-normalized sequence score
print(mlm_model.sequence_score(stimuli, reduction = lambda x: -x.sum(0).item(), PLL_metric='within_word_l2r'))
'''
[32.77983617782593]
'''

# original metric, for comparison:
print(mlm_model.sequence_score(stimuli, reduction = lambda x: -x.sum(0).item(), PLL_metric='original'))
'''
[18.014726161956787]
'''

print(mlm_model.token_score(stimuli, PLL_metric='within_word_l2r'))
'''
[[('the', -0.07324600219726562), ('traveler', -9.668401718139648), ('lost', -6.955361366271973),
('the', -1.1923179626464844), ('so', -7.776356220245361), ('##uven', -6.989711761474609),
('##ir', -0.037807464599609375), ('.', -0.08663368225097656)]]
'''

# original values, for comparison (notice the 'souvenir' tokens):

print(mlm_model.token_score(stimuli, PLL_metric='original'))
'''
[[('the', -0.07324600219726562), ('traveler', -9.668402671813965), ('lost', -6.955359935760498), ('the', -1.192317008972168), ('so', -3.0517578125e-05), ('##uven', -0.0009250640869140625), ('##ir', -0.03780937194824219), ('.', -0.08663558959960938)]]
'''
```

## NEW: Vision-Language Model (VLM) Scoring

Minicons now supports VLM scoring! The following code demonstrates how one can extract log-probs of caption/descriptions from Salesforce's BLIP-2 model, conditioned on a batch of images:

<img align="right" src="assets/vlminicons.png" width="350px">


```py
from minicons import scorer
from PIL import Image

# top image
penguin = Image.open('penguin.jpg')

# bottom image
cardinal = Image.open('cardinal.jpg')

lm = scorer.VLMScorer(
  "Salesforce/blip2-opt-2.7b", 
  device="cuda:0"
)

lm.sequence_score(
  text_batch=["This bird can fly."] * 2, 
  image_batch=[penguin, cardinal]
)

#> logprobs of penguin vs cardinal -> can fly
#> [-5.644123077392578, -5.129026889801025]
```

## OpenAI API

> [!CAUTION]
> THIS IS NOW DEPRECATED BECAUSE OPEN-AI NO LONGER MAKES INPUT LOGPROBS AVAILABLE!**

Some models on the OpenAI API also allow for querying of log-probs (for now), and minicons now (as of Sept 29) also supports it! Here's how:

First, make sure you save your OpenAI API Key in some file (say `~/.openaikey`). Register the key using:
```py
from minicons import openai as mo

PATH = "/path/to/apikey"
mo.register_api_key(PATH)
```
Then,

```py
from minicons import openai as mo

stimuli = ["the keys to the cabinet are", "the keys to the cabinet is"]

# we want to test if p(are | prefix) > p(is | prefix)
model = "gpt-3.5-turbo-instruct"
query = mo.OpenAIQuery(model, stimuli)

# run query using the above batch
query.query()

# get conditional log-probs for are and is given prior context:
query.conditional_score(["are", "is"])

#> [-2.5472614765167236, -5.633198261260986] SUCCESS!

# NOTE: this will not be 100% reproducible since it seems OpenAI adds a little noise to its outputs.
# see https://twitter.com/xuanalogue/status/1653280462935146496
```

## Tutorials

- [Introduction to using LM-scoring methods using minicons](https://kanishka.website/post/minicons-running-large-scale-behavioral-analyses-on-transformer-lms/)
- [Computing sentence and token surprisals using minicons](examples/surprisals.md)
- [Extracting word/phrase representations using minicons](examples/word_representations.md)

## Recent Updates
- **November 6, 2021:** MLM scoring has been fixed! You can now use `model.token_score()` and `model.sequence_score()` with `MaskedLMScorers` as well!
- **June 4, 2022:** Added support for Seq2seq models. Thanks to [Aaron Mueller](https://github.com/aaronmueller) 🥳
- **June 13, 2023:** Added support for `within_word_l2r`, a better way to do MLM scoring, thanks to Carina Kauf (https://github.com/carina-kauf) 🥳
- **January, 2024:** minicons now supports mamba!

## Citation

If you use `minicons`, please cite the following paper:

```tex
@article{misra2022minicons,
    title={minicons: Enabling Flexible Behavioral and Representational Analyses of Transformer Language Models},
    author={Kanishka Misra},
    journal={arXiv preprint arXiv:2203.13112},
    year={2022}
}
```

If you use Kauf and Ivanova's PLL scoring technique, please additionally also cite the following paper:

```tex
@inproceedings{kauf2023better,
  title={A Better Way to Do Masked Language Model Scoring},
  author={Kauf, Carina and Ivanova, Anna},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year={2023}
}
```

## Famous users of minicons:

A non-exhaustive but fun list of ppl:

* Adele Goldberg
* Chris Potts
* Najoung Kim
* Forrest Davis
* Marten van Schijndel
* Valentina Pyatkin
* Aaron Mueller
* Sanghee Kim
* Venkata Govindarajan
* Kyle Mahowald
* Carina Kauf

