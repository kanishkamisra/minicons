## Calculating surprisals with transformer models using minicons

This brief document shows how one can calculate surprisals for sentences using models such as `gpt` and `gpt2`. 

For demonstration purposes I will use `gpt2`(small) from Huggingface, and evaluate it on a number agreement task from the [BLiMP dataset](https://github.com/alexwarstadt/blimp/). This task specifically tests whether the model assigns greater probability to "hasn't" as compared to "haven't" in pairs of stimuli such as (1) and (2):

(1) The sketch of those trucks hasn't 

(2) The sketch of those trucks haven't

Converting this into a hypothesis dealing with surprisals, the model should therefore be "more surprised" to see (2) than (1).

`minicons` helps in performing such experiments:

```py
import minicons
import torch
from torch.utils.data import DataLoader

import json
```
Incremental models can be instantiated using:
```py
# Warning: This will download a 550mb model file if you do not already have it!
model = scorer.IncrementalLMScorer('gpt2', 'cpu')
```

`minicons` allows you to compute token-by-token log-probabilities using the `model.logprobs()` function, which accepts texts encoded by the `model.prepare_text()` function. 

```py
logprobs = model.logprobs(model.prepare_text("The sketch of those trucks hasn't"))

print(logprobs)

# [(tensor([ -7.6899, -10.9180,  -2.3835,  -5.1761, -12.6967, -12.9571,  -9.4133, -0.0216], dtype=torch.float64), ['ĠThe', 'Ġsketch', 'Ġof', 'Ġthis', 'e', 'Ġtrucks', 'Ġhasn', "'t"])]
```

Note that you can also pass a batch of texts in a list format.

```py
logprobs = model.logprobs(model.prepare_text(["The sketch of those trucks hasn't", "The sketch of those trucks haven't"]))

print(logprobs)

# [(tensor([-7.6899e+00, -1.0918e+01, -2.3835e+00, -7.2515e+00, -8.8136e+00, -8.7238e+00, -7.0190e-04], dtype=torch.float64), ['ĠThe', 'Ġsketch', 'Ġof', 'Ġthose', 'Ġtrucks', 'Ġhasn', "'t"]), (tensor([-7.6899e+00, -1.0918e+01, -2.3835e+00, -7.2515e+00, -8.8136e+00, -1.0561e+01, -1.3885e-03], dtype=torch.float64), ['ĠThe', 'Ġsketch', 'Ġof', 'Ġthose', 'Ġtrucks', 'Ġhaven', "'t"])]

# Convert log-probabilities into token-by-token surprisals:

surprisals = [-x for x in list(zip(*logprobs))[0]]

print(surprisals)

# [tensor([7.6899e+00, 1.0918e+01, 2.3835e+00, 7.2515e+00, 8.8136e+00, 8.7238e+00, 7.0190e-04], dtype=torch.float64), tensor([7.6899e+00, 1.0918e+01, 2.3835e+00, 7.2515e+00, 8.8136e+00, 1.0561e+01, 1.3885e-03], dtype=torch.float64)]
```

You can also compute the overall sentence scores by using the `model.score()` function. By default it does so by normalizing the summed log probability score and dividing it by the length. To only get the overall log-probability, one would pass `pool = torch.sum` as an argument:

```py
model.score(["The sketch of those trucks hasn't", "The sketch of those trucks haven't"], pool = torch.sum)

# Log probabilities of the sentences:
# [-45.78099822998047, -47.618995666503906]
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
    good_scores.extend(model.score(good), pool = torch.sum)
    bad_scores.extend(model.score(bad), pool = torch.sum)


# Testing the extent to which GPT2-small shows patterns of number-agreement:
print(np.mean([g > b for g,b in zip(good_scores, bad_scores)]))

# 0.804
```