from typing import *
from collections import defaultdict
import argparse
import os
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from ..scorer import IncrementalLMScorer, MaskedLMScorer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Utility to quickly retrieve per-token score for sentences using language models."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument(
        "--scorer",
        type=str,
        required=True,
        choices=['incremental', 'masked'],
        help="Language model type: either incremental or masked."
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Name of the model (can use any model from the huggingface model hub https://huggingface.co/models)."
    )

    input_group.add_argument(
        "--text",
        type=str,
        help="Single sentence to score. Use --file instead for scoring multiple sentences."
    )

    input_group.add_argument(
        "--file",
        type=str,
        help="File containing one sentence per line."
    )

    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=1,
        help="Number of sentences to process in parallel."
    )

    parser.add_argument(
        "--num-workers",
        "-nw",
        type=int,
        default=2,
        help="Number of workers to spawn for scoring the sentences in the input file."
    )

    parser.add_argument(
        "--prob",
        action="store_true",
        help="Get per-token probabilities for every sentence in the input."
    )

    # parser.add_argument(
    #     "--logprob",
    #     "log-probability",
    #     default=True,
    #     action="store_true",
    #     help="Get per-token log probabilities for every sentence in the input."
    # )

    parser.add_argument(
        "--surprisal",
        action="store_true",
        help="Get per-token surprisals for every sentence in the input."
    )

    parser.add_argument(
        "--rank",
        action="store_true",
        help="Get per-token ranks (scored by log-probability) for every sentence in the input."
    )

    parser.add_argument(
        "--base-two",
        "--in_bits",
        action="store_true"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="If provided, it runs the model on the specified cuda device."
    )

    return parser.parse_args()

def normalize_args(args: argparse.Namespace) -> None:
    if (args.file):
        args.file = os.path.realpath(args.file)

def validate_args(args: argparse.Namespace) -> None:
    if (args.file):
        if (not os.path.isfile(args.file)):
            raise ValueError("The provided sentences file path is invalid.")
        
        if (args.batch_size == 1 or args.num_workers == 2):
            print('Currently using batch size of {} and {} workers. Please configure them according to your hardware.'.format(args.batch_size, args.num_workers))
    
    if (args.device != 'cpu'):
        if (not torch.cuda.is_available()):
            raise ValueError("Cuda not available.")
        if (int(args.device.split(':')[-1]) >= torch.cuda.device_count()):
            raise ValueError("Invalid device.")

def pretty_print(results: Iterable) -> None:
    print(results.round(2).to_string(index=False))

def cli_token_score(self, batch, model_type, surprisal = False, prob = False, base_two = False, rank = False, batch_idx = 0, global_batch_size = 1):
    
    result = defaultdict(list)

    tokenized = self.prepare_text(batch)
    if rank:
        scores, ranks = self.compute_stats(tokenized, rank = rank, base_two = base_two, return_tensors=True)
    else:
        scores = self.compute_stats(tokenized, base_two = base_two, return_tensors=True)

    scores = [s.tolist() for s in scores]

    if model_type == 'incremental':
        indices = [[i for i in indexed if i != self.tokenizer.pad_token_id] for indexed in tokenized[0]['input_ids'].tolist()]
    else:
        indices = [[i.item() for i in indexed if i.item() != self.tokenizer.pad_token_id] for indexed in list(zip(*tokenized))[2]]

    tokens = [self.decode(idx) for idx in indices]

    if rank:
        for i, (t, s, r) in enumerate(zip(tokens, scores, ranks)):
            sent_id = i + batch_idx * global_batch_size + 1
            diff = 0
            if len(t) > len(s):
                diff = len(t) - len(s)
                s = [None]*diff + s
                r = [None]*diff + r
            result['sentence_id'].extend([sent_id]*len(t))
            result['token_id'].extend([j+1 for j, _ in enumerate(t)])
            result['token'].extend(t)
            result['logprob'].extend(s)
            
            if prob:
                probs = [None] * diff + torch.tensor(s[diff:]).exp().tolist()
                result['prob'].extend(probs)
            
            if surprisal:
                surps = [-1.0 * x if x is not None else x for x in s]
                result['surprisal'].extend(surps)

            result['rank'].extend(r)
    else:
        for i, (t, s) in enumerate(zip(tokens, scores)):
            sent_id = i + batch_idx * global_batch_size + 1
            diff = 0
            if len(t) > len(s):
                diff = len(t) - len(s)
                s = [None]*diff + s
            result['sentence_id'].extend([sent_id]*len(t))
            result['token_id'].extend([j+1 for j, _ in enumerate(t)])
            result['token'].extend(t)
            result['logprob'].extend(s)
            
            if prob:
                probs = [None] * diff + torch.tensor(s[diff:]).exp().tolist()
                result['prob'].extend(probs)
            
            if surprisal:
                surps = [-1.0 * x if x is not None else x for x in s]
                result['surprisal'].extend(surps)
    return result

def main(args: argparse.Namespace):
    modeltype = args.scorer.lower()
    if (modeltype == 'incremental'):
        lm = IncrementalLMScorer(args.model, args.device)
    elif (modeltype == 'masked'):
        lm = MaskedLMScorer(args.model, args.device)
    else:
        raise Exception("Incorrect scorer passed. Use either incremental or masked.")
    
    if (args.text):
        processed_results = cli_token_score(lm, args.text, modeltype, args.surprisal, args.prob, args.base_two, args.rank)
    else:
        sentence_file = open(args.file, 'r')
        sentences = [sentence.strip() for sentence in sentence_file]
        sentence_dl = DataLoader(sentences, batch_size=args.batch_size, num_workers=args.num_workers)
        results = []
        for i, batch in enumerate(tqdm(sentence_dl)):
            results.append(cli_token_score(lm, batch, modeltype, args.surprisal, args.prob, args.base_two, args.rank, batch_idx = i, global_batch_size=args.batch_size))
        
        processed_results = defaultdict(list)
        for result in results:
            for k, v in result.items():
                processed_results[k].extend(v)

    processed_results = pd.DataFrame(processed_results)
    pretty_print(processed_results)

def process():
    try:
        args = parse_args()
        validate_args(args)
        main(args)
    except KeyboardInterrupt:
        print('\nProcess stopped by user.')
    except Exception as er:
        print('Error: {}'.format(er))

if __name__ == "__main__":
    process()