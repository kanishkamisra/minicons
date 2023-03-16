## Calculating surprisals with transformer models using minicons

This brief document shows how one can calculate surprisals for sentences using models such as `gpt` and `gpt2`. 

For demonstration purposes I will use `gpt2`(small) from Huggingface, and evaluate it on a number agreement task from the [BLiMP dataset](https://github.com/alexwarstadt/blimp/). This task specifically tests whether the model assigns greater probability to "hasn't" as compared to "haven't" in pairs of stimuli such as (1) and (2):

(1) The sketch of those trucks hasn't 

(2) The sketch of those trucks haven't

Converting this into a hypothesis dealing with surprisals, the model should therefore be "more surprised" to see (2) than (1).

`minicons` helps in performing such experiments:

```py
from minicons import scorer
import torch
from torch.utils.data import DataLoader

import numpy as np

import json
```
Incremental models can be instantiated using:
```py
# Warning: This will download a 550mb model file if you do not already have it!
model = scorer.IncrementalLMScorer('gpt2', 'cpu')
```

`minicons` allows you to compute token-by-token log-probabilities using the `model.compute_stats()` function, which accepts texts encoded by the `model.prepare_text()` function. It has the following parameters:

```
batch [Iterable]: Input batch (list of sentences or single sentence)
rank [bool]: Whether the model should return ranks of each token (by probability)
base_two [bool]: Use base 2 for the log-prob
return_tensors [bool]: Whether the output should contain tensors.
```

Each value here represents the conditional probability -- P(word | left context), so the first value represents the probability of the second word given the first.

```py
logprobs = model.compute_stats(model.prepare_text("The sketch of those trucks hasn't"))

print(logprobs)

#[[-10.879678726196289, -2.5105514526367188,  -6.6631927490234375,  -8.962379455566406,  -8.681724548339844,  -0.0005340576171875]]
```

Note that you can also pass a batch of texts in a list format.

```py
sentences = ["The sketch of those trucks hasn't", "The sketch of those trucks haven't"]

model.compute_stats(model.prepare_text(sentences))

# [[-10.879678726196289,
#  -2.5105514526367188,
#  -6.6631927490234375,
#  -8.962379455566406,
#  -8.681724548339844,
#  -0.0005340576171875],
# [-10.879678726196289,
#  -2.5105514526367188,
#  -6.6631927490234375,
#  -8.962379455566406,
#  -10.669326782226562,
#  -0.0013275146484375]]
```
To also get tokens in the output, use the following code. Note: `minicons` adds an additional `0.0` log-probability for the first token/word as convention.

```py
model.token_score(sentences)

'''
[[('The', 0.0),
  ('sketch', -10.879678726196289),
  ('of', -2.5105514526367188),
  ('those', -6.6631927490234375),
  ('trucks', -8.962379455566406),
  ('hasn', -8.681724548339844),
  ("'t", -0.0005340576171875)],
 [('The', 0.0),
  ('sketch', -10.879678726196289),
  ('of', -2.5105514526367188),
  ('those', -6.6631927490234375),
  ('trucks', -8.962379455566406),
  ('haven', -10.669326782226562),
  ("'t", -0.0013275146484375)]]
'''
```

For surprisals, pass `surprisal = True` to `model.token_score()` (pass `base_two = True` if you want surprisals in bits)

```py
model.token_score(sentences, surprisal = True, base_two = True)

'''
[[('The', 0.0),
  ('sketch', 15.69605827331543),
  ('of', 3.621960163116455),
  ('those', 9.612955093383789),
  ('trucks', 12.929980278015137),
  ('hasn', 12.525080680847168),
  ("'t", 0.0007704822928644717)],
 [('The', 0.0),
  ('sketch', 15.69605827331543),
  ('of', 3.621960163116455),
  ('those', 9.612955093383789),
  ('trucks', 12.929980278015137),
  ('haven', 15.392584800720215),
  ("'t", 0.0019151987507939339)]]
'''
```

You can also compute the overall sentence scores by using the `model.sequence_score()` function. By default it does so by normalizing the summed log probability score and dividing it by the length. To only get the overall log-probability, one would pass `reduction = lambda x: x.sum(0)` (for surprisals pass `lambda x: -x.sum(0)`) as an argument:

```py
model.sequence_score(["The sketch of those trucks hasn't", "The sketch of those trucks haven't"], reduction = lambda x: x.sum(0))

# Log probabilities of the sentences:
# [tensor(-37.6981), tensor(-39.6865)]
```

Finally, `minicons` also facilitates large-scale experiments. For example, let's run our test of GPT2-small's behavior on the full number-agreement task from BLiMP:

```py
stimuli = []
with open("distractor_agreement_relational_noun.jsonl", "r") as f:
    for line in f:
        row = json.loads(line)
        stimuli.append([row['one_prefix_prefix'] + " " + row['one_prefix_word_good'], row['one_prefix_prefix'] + " " + row['one_prefix_word_bad']])

for pair in stimuli[:5]:
    print(f"{pair[0]} vs. {pair[1]}")

## A niece of most senators hasn't vs. A niece of most senators haven't
## The sketch of those trucks hasn't vs. The sketch of those trucks haven't
## A newspaper article about the Borgias has vs. A newspaper article about the Borgias have
## The niece of most guests has vs. The niece of most guests have
## A sketch of lights doesn't vs. A sketch of lights don't

stimuli_dl = DataLoader(stimuli, batch_size = 100)

good_scores = []
bad_scores = []
for batch in stimuli_dl:
    good, bad = batch
    good_scores.extend(model.sequence_score(good), reduction = lambda x: x.sum(0))
    bad_scores.extend(model.sequence_score(bad), reduction = lambda x: x.sum(0))


# Testing the extent to which GPT2-small shows patterns of number-agreement:
print(np.mean([g > b for g,b in zip(good_scores, bad_scores)]))

# 0.804
```
