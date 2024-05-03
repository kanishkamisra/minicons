Here's a quick demonstration of what I think is the core use of minicons:

```py
from minicons import scorer

lm = scorer.IncrementalLMScorer('EleutherAI/pythia-1.4b', 'cuda:0')

# batched sentence log-probs/token
stimuli = ["The keys to the cabinet are on the table.",
           "The keys to the cabinet is on the table."]

print(lm.sequence_score(stimuli))

#> [-3.2771975994110107, -3.968510627746582]

# token-wise log-probs (ignores first token)

print(lm.token_score(stimuli, rank=True)) # log-probs + ranks per token in output distribution
'''
[[('The', 0.0, 0),
  ('keys', -9.773914337158203, 2764),
  ('to', -1.6787223815917969, 2),
  ('the', -1.957301139831543, 1),
  ('cabinet', -7.117016315460205, 127),
  ('are', -1.2137947082519531, 1),
  ('on', -3.597787857055664, 4),
  ('the', -0.4375495910644531, 1),
  ('table', -2.5509910583496094, 2),
  ('.', -1.1676998138427734, 1)],
 [('The', 0.0, 0),
  ('keys', -9.773914337158203, 2764),
  ('to', -1.6787223815917969, 2),
  ('the', -1.957301139831543, 1),
  ('cabinet', -7.117016315460205, 127),
  ('is', -5.006587982177734, 15),
  ('on', -5.69062614440918, 24),
  ('the', -0.2875690460205078, 1),
  ('table', -2.921924591064453, 4),
  ('.', -1.2829303741455078, 1)]]
'''

# Conditional Scoring

prefixes = ["a robin", "a penguin"]
targets = ["can fly."] * 2

print(lm.conditional_score(prefixes, targets)) # log-prob/token of target phrase

#> [-4.6671142578125, -4.162600517272949] # guess the model doesn't know many things about robins (relative to penguins)
```
