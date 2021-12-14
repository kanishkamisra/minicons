from typing import *

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
        "--num_workers",
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
        "--base_two",
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
    print(results.to_string(index=False))

def main(args: argparse.Namespace):
    if (args.scorer.lower() == 'incremental'):
        lm = IncrementalLMScorer(args.model, args.device)
    elif (args.scorer.lower() == 'masked'):
        lm = MaskedLMScorer(args.model, args.device)
    else:
        raise Exception("Incorrect scorer passed. Use either incremental or masked.")
    
    if (args.text):
        results = lm.token_score(args.text, args.surprisal, args.prob, args.base_two, args.rank)
    else:
        sentence_file = open(args.file, 'r')
        sentences = [sentence.strip() for sentence in sentence_file]
        sentence_dl = DataLoader(sentences, batch_size=args.batch_size, num_workers=args.num_workers)
        results = []
        for batch in tqdm(sentence_dl):
            results.extend(lm.token_score(batch, args.surprisal, args.prob, args.base_two, args.rank))
    
    processed_results = []
    for sentence_ind, sentence in enumerate(results):
        for score_ind, score in enumerate(sentence):
            processed_results.append([sentence_ind + 1, score_ind + 1] + list(score))

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