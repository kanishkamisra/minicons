from minicons import scorer

lm = scorer.IncrementalLMScorer("gpt2-xl", "cuda:0") # change to cpu if you can.

stimuli = ["I was a matron in France.", "I was a mat in France."]

# old way, no correction
# P.S. gpt2 does not automatically add a bos token at the beginning...
lm.token_score(stimuli, bos_token=True, surprisal=True, base_two=True, bow_correction=False)

'''
[[('<|endoftext|>', 0.0),
  ('I', 5.85346794128418),
  ('was', 4.284247398376465),
  ('a', 4.673775672912598),
  ('mat', 16.344608306884766),
  ('ron', 1.7366491556167603),
  ('in', 2.1214966773986816),
  ('France', 11.425250053405762),
  ('.', 3.4500668048858643)],
 [('<|endoftext|>', 0.0),
  ('I', 5.85346794128418),
  ('was', 4.284247398376465),
  ('a', 4.673775672912598),
  ('mat', 16.344608306884766),
  ('in', 10.782632827758789),
  ('France', 10.712746620178223),
  ('.', 3.0245020389556885)]]
'''

# the new way! notice the surprisal of "mat" in both cases
lm.token_score(stimuli, bos_token=True, surprisal=True, base_two=True, bow_correction=True)

'''
[[('<|endoftext|>', 0.0),
  ('I', 6.3007493019104),
  ('was', 3.8401684761047363),
  ('a', 4.676173686981201),
  ('mat', 16.339006423950195),
  ('ron', 2.110022783279419),
  ('in', 1.7500823736190796),
  ('France', 11.423291206359863),
  ('.', 3.6690571308135986)],
 [('<|endoftext|>', 0.0),
  ('I', 6.3007493019104),
  ('was', 3.8401684761047363),
  ('a', 4.676173686981201),
  ('mat', 21.335359573364258),
  ('in', 5.801166534423828),
  ('France', 10.697859764099121),
  ('.', 3.333509683609009)]]
'''