Scorer
======

Classes for scoring sequences using language models. The main entry points are
:class:`~minicons.scorer.MaskedLMScorer` for BERT-style models,
:class:`~minicons.scorer.IncrementalLMScorer` for autoregressive models (GPT-2, Llama, etc.),
and :class:`~minicons.scorer.VLMScorer` for vision-language models.

.. toctree::
   :maxdepth: 1

   scorer/MaskedLMScorer
   scorer/IncrementalLMScorer
   scorer/Seq2SeqScorer
   scorer/MambaScorer
   scorer/VLMScorer
