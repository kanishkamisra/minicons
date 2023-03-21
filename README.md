# minicons: Enabling Flexible Behavioral and Representational Analyses of Transformer Language Models

[![Downloads](https://static.pepy.tech/personalized-badge/minicons?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/minicons)

This repo is a wrapper around the `transformers` [library](https://huggingface.co/transformers) from hugging face :hugs:

<!-- TODO: Description-->



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

# if model is seq2seq:
model = cwe.EncDecCWE('t5-small')

print(model.extract_representation(context_words))

'''(last layer, by default)
tensor([[-0.0895,  0.0758,  0.0753,  ...,  0.0130, -0.1093, -0.2354],
        [-0.0695,  0.1142,  0.0803,  ...,  0.0807, -0.1139, -0.2888]])
'''
```

2. Compute sentence acceptability measures (surprisals) using Word Prediction Models:

```py
from minicons import scorer

mlm_model = scorer.MaskedLMScorer('bert-base-uncased', 'cpu')
ilm_model = scorer.IncrementalLMScorer('distilgpt2', 'cpu')
s2s_model = scorer.Seq2SeqScorer('t5-base', 'cpu')

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

# Seq2seq scoring
## Blank source sequence, target sequence specified in `stimuli`
print(s2s_model.sequence_score(stimuli, source_format = 'blank'))
## Source sequence is the same as the target sequence in `stimuli`
print(s2s_model.sequence_score(stimuli, source_format = 'copy'))
'''
[-7.910910129547119, -7.835635185241699]
[-10.555519104003906, -9.532546997070312]
'''
```

## Tutorials

- [Introduction to using LM-scoring methods using minicons](https://kanishka.xyz/post/minicons-running-large-scale-behavioral-analyses-on-transformer-lms/)
- [Computing sentence and token surprisals using minicons](examples/surprisals.md)
- [Extracting word/phrase representations using minicons](examples/word_representations.md)

## Recent Updates
- **November 6, 2021:** MLM scoring has been fixed! You can now use `model.token_score()` and `model.sequence_score()` with `MaskedLMScorers` as well!
- **June 4, 2022:** Added support for Seq2seq models. Thanks to [Aaron Mueller](https://github.com/aaronmueller) 🥳

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
