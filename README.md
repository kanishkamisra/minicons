# minicons

[![Downloads](https://static.pepy.tech/personalized-badge/minicons?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/minicons)

Helper functions for analyzing Transformer based representations of language

This repo is a wrapper around the `transformers` [library](https://huggingface.co/transformers) from hugging face :hugs:


## Installation

Install from Pypi using:

```pip install minicons```

## Supported Functionality

- Extract word representations from Contextualized Word Embeddings
- Score sequences using language model scoring techniques, including masked language models following [Salazar et al. (2020)](https://www.aclweb.org/anthology/2020.acl-main.240.pdf).


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
```

2. Compute sentence acceptability measures (surprisals) using Incremental and Masked Language Models:

```py
from minicons import scorer

mlm_model = scorer.MaskedLMScorer('bert-base-uncased', 'cpu')
ilm_model = scorer.IncrementalLMScorer('distilgpt2', 'cpu')

stimuli = ["The keys to the cabinet are on the table.",
           "The keys to the cabinet is on the table."]

print(mlm_model.score(stimuli, pool = torch.sum))

'''
[13.962650299072266, 23.41507911682129]
'''

print(ilm_model.score(stimuli, pool = torch.sum))

'''
[41.51601982116699, 44.497480392456055]
'''
```

## Tutorials

- [Introduction to using LM-scoring methods using minicons](https://kanishka.xyz/post/minicons-running-large-scale-behavioral-analyses-on-transformer-lms/)
- [Computing Surprisals using minicons](examples/surprisals.md)

## Upcoming features:

- Explore attention distributions extracted from transformers.
- Contextual cosine similarities, i.e., compute a word's cosine similarity with every other word in the input context with batched computation.
- Open to suggestions!
