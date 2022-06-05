# Extracting Word and Phrase Representations using minicons

`minicons` allows for efficient and intuitive extraction of word/phrase representations using transformer models (in theory, any model that is available on the huggingface hub).
It does so by using a wrapper (in the `cwe` module) around the `AutoModel` class made available by the transformers package by HuggingFace.

For demonstration purposes, let's use the `bert-base-uncased` model to extract contextual representations of the word *aircraft* from the list of sentences in the file `samplesentences.txt` (listed in the same directory as this file).

## Preliminaries

### Required packages
This tutorial requires the following packages:

```
minicons==0.1.15
torch>=1.8.0
```

### Importing libraries

```py
import torch
from minicons import cwe 
from minicons.utils import character_span # for demonstrating.
```

### Loading sentences
We will now load our sentences, each of which consists of the target word *aircraft*:
```py
sentences = []
with open('samplesentences.txt', 'r') as f:
    for line in f:
        sentences.append(line.strip())

sentences[:2]

'''
['This aircraft works by jet propulsion.',
 'His passion is making model aircraft.']
'''

len(sentences)

'''
11
'''
```

### Loading the model
Every representation extraction model in `minicons` is an instance of the `cwe.CWE` class. CWE stands for 'contextual word embeddings.'

```py
model = cwe.CWE('bert-base-uncased', device = 'cpu') # also allows gpus, use 'cuda:[NUMBER]' to do so.
```

## Representation Extraction
The function primarily used for extracting representations from models is `model.extract_representation()`. It accepts batches of instances represented in either of the following formats:

```
data = [
  (sentence_1, word_1),
  (sentence_2, word_2),
  ....
  (sentence_n, word_n)
]
```
or

```
data = [
  (sentence_1, (start_1, end_1)),
  (sentence_2, (start_2, end_2)),
  ....
  (sentence_n, (start_n, end_n))
]
```
where `(start_i, end_i)` are the character span indices for the target word in the ith sentence, i.e., `start_i` is the start index, and `end_i` is the end index.

For example, the instance `["I like reading books.", (15, 20)]` corresponds to the word `"books"`.

Regardless of what is specified, `extract_representation()` reduces the input to the second format. For instance, to get the character span indices of *aircraft* in the first sentence:

```py
character_span(sentences[0], 'aircraft')

'''
(5, 13)
'''
```

The first type indeed provides an easier method to prepare our data to extract representations:

```py
instances = []
for s in sentences:
    instances.append([s, 'aircraft'])
```

We can now use `instances` as an input to `model.extract_representation()`. By default, this method extracts representations from the last layer of the model:

```py
model.extract_representation(instances)

'''
tensor([[ 0.4418,  0.2876, -0.4887,  ..., -0.9296,  0.2270,  0.4052],
        [ 0.6424,  0.3509,  0.1863,  ..., -0.7924,  0.0086, -0.5835],
        [-0.1713, -0.0228, -0.1329,  ..., -0.8026,  0.6910,  0.0614],
        ...,
        [ 0.1711,  0.3773, -1.2992,  ..., -0.3187,  0.3004, -0.4013],
        [ 0.6711,  0.0556, -1.1710,  ..., -0.5866,  0.8467,  0.2816],
        [ 0.5522,  0.1332,  0.2180,  ..., -0.2806,  1.0105, -0.1799]])
'''
```
We can even choose a different layer:

```py
model.extract_representation(instances, layer = 5)

'''
tensor([[ 1.1305,  1.2379, -0.3605,  ..., -0.2164,  0.7634,  0.2290],
        [ 1.5314,  1.1103, -0.3012,  ...,  0.3013,  1.1243, -0.1035],
        [ 0.5519,  0.3571,  0.1852,  ..., -0.0317,  0.3467, -0.5793],
        ...,
        [ 0.2921,  0.5046, -0.8121,  ..., -0.0687,  0.5321,  0.0470],
        [ 0.6099,  0.9541, -0.5739,  ...,  0.1725,  0.2572, -0.0846],
        [ 0.8634,  0.3579, -0.1031,  ..., -0.4277, -0.0830, -0.3293]])
'''
```

or even specify multiple layers, which yields a list of torch tensors:

```py
model.extract_representation(instances, layer = [11, 12])

'''
[tensor([[ 0.9413,  0.3149,  0.0279,  ..., -1.2449,  0.5606, -0.0098],
         [ 1.2197,  0.4842,  0.0792,  ..., -1.3511,  0.3262, -0.8011],
         [ 0.1097, -0.0434,  0.4395,  ..., -1.2032,  1.0695,  0.1302],
         ...,
         [ 0.5841,  0.3509, -0.4591,  ..., -0.2502,  0.5510, -0.2269],
         [ 0.8643,  0.2670, -0.8616,  ..., -0.4080,  0.3723, -0.0150],
         [ 0.9704,  0.2731,  0.0032,  ..., -0.4349,  1.3013, -0.1415]]),
 tensor([[ 0.4418,  0.2876, -0.4887,  ..., -0.9296,  0.2270,  0.4052],
         [ 0.6424,  0.3509,  0.1863,  ..., -0.7924,  0.0086, -0.5835],
         [-0.1713, -0.0228, -0.1329,  ..., -0.8026,  0.6910,  0.0614],
         ...,
         [ 0.1711,  0.3773, -1.2992,  ..., -0.3187,  0.3004, -0.4013],
         [ 0.6711,  0.0556, -1.1710,  ..., -0.5866,  0.8467,  0.2816],
         [ 0.5522,  0.1332,  0.2180,  ..., -0.2806,  1.0105, -0.1799]])]
'''

# These can be pooled using the following code:
torch.stack(model.extract_representation(instances, layer = [11, 12])).mean(0)

'''
tensor([[ 0.6916,  0.3012, -0.2304,  ..., -1.0873,  0.3938,  0.1977],
        [ 0.9310,  0.4175,  0.1328,  ..., -1.0717,  0.1674, -0.6923],
        [-0.0308, -0.0331,  0.1533,  ..., -1.0029,  0.8803,  0.0958],
        ...,
        [ 0.3776,  0.3641, -0.8791,  ..., -0.2845,  0.4257, -0.3141],
        [ 0.7677,  0.1613, -1.0163,  ..., -0.4973,  0.6095,  0.1333],
        [ 0.7613,  0.2032,  0.1106,  ..., -0.3577,  1.1559, -0.1607]])
'''
```

A small trick to extract from the last 4 layers of the model involves using the `model.layers` attribute.

```py
last_four = list(range(model.layers+1))[-4:]
model.extract_representation(instances, layer = last_four)

'''
[tensor([[ 0.9946,  0.9216, -0.2156,  ..., -0.4210,  0.3931,  0.2772],
         [ 1.3800,  0.7844, -0.3833,  ..., -0.2114,  0.5890, -0.4240],
         [ 0.2445,  0.2528,  0.3336,  ..., -0.4638,  0.3499, -0.2713],
         ...,
         [ 0.6907,  0.4878, -0.2212,  ...,  0.0482,  0.0307,  0.0583],
         [ 0.6474,  0.9128, -0.1065,  ..., -0.6336, -0.1677, -0.2797],
         [ 1.0601,  0.2845,  0.2318,  ..., -0.7379,  0.6266, -0.3281]]),
 tensor([[ 1.0311,  0.2918,  0.0645,  ..., -1.2735,  0.6836, -0.3382],
         [ 1.3628,  0.4729, -0.1582,  ..., -0.9663,  0.5652, -0.9539],
         [ 0.3337, -0.0796,  0.3472,  ..., -0.8018,  0.7095, -0.5963],
         ...,
         [ 0.8667,  0.5114, -0.4021,  ..., -0.2998,  0.3388, -0.5384],
         [ 0.5956,  0.2355, -0.1907,  ..., -0.3381,  0.0460, -0.2902],
         [ 1.1451,  0.1151,  0.1154,  ..., -0.5127,  1.3450, -0.5618]]),
 tensor([[ 0.9413,  0.3149,  0.0279,  ..., -1.2449,  0.5606, -0.0098],
         [ 1.2197,  0.4842,  0.0792,  ..., -1.3511,  0.3262, -0.8011],
         [ 0.1097, -0.0434,  0.4395,  ..., -1.2032,  1.0695,  0.1302],
         ...,
         [ 0.5841,  0.3509, -0.4591,  ..., -0.2502,  0.5510, -0.2269],
         [ 0.8643,  0.2670, -0.8616,  ..., -0.4080,  0.3723, -0.0150],
         [ 0.9704,  0.2731,  0.0032,  ..., -0.4349,  1.3013, -0.1415]]),
 tensor([[ 0.4418,  0.2876, -0.4887,  ..., -0.9296,  0.2270,  0.4052],
         [ 0.6424,  0.3509,  0.1863,  ..., -0.7924,  0.0086, -0.5835],
         [-0.1713, -0.0228, -0.1329,  ..., -0.8026,  0.6910,  0.0614],
         ...,
         [ 0.1711,  0.3773, -1.2992,  ..., -0.3187,  0.3004, -0.4013],
         [ 0.6711,  0.0556, -1.1710,  ..., -0.5866,  0.8467,  0.2816],
         [ 0.5522,  0.1332,  0.2180,  ..., -0.2806,  1.0105, -0.1799]])]
'''
```

### Extracting reprsentations of phrases

One can even pass constituent phrases of a sentence in each instance to extract representations of phrases (by default they are pooled using an average). For instance:

```py
phrases = [
    ['I like reading books.', 'reading books'], 
    ['I also like riding my bike!', 'riding my bike']
]

model.extract_representation(phrases)

'''
tensor([[ 0.2985,  0.6786,  0.2250,  ...,  0.1723,  0.3650, -0.5355],
        [ 1.2477,  0.1224, -0.0942,  ..., -0.0835, -0.2701, -0.2143]])
'''
```

Fin.