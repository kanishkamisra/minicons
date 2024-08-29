import random
import re
import string
import torch

from itertools import groupby
from typing import List, Tuple, Optional
from wonderwords import (
    RandomWord,
)  # Basic library to randomly generate lists of words


def get_batch(data: list, batch_size: int, shuffle: bool = False):
    if shuffle:
        random.shuffle(data)
    sindex = 0
    eindex = batch_size
    while eindex < len(data):
        batch = data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(data):
        batch = data[sindex:]
        yield batch


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def between(num, tup):
    if num >= tup[0] and num < tup[1]:
        return True
    else:
        return False


def character_span(sentence, word):
    assert word in sentence
    idx = sentence.find(word)
    return idx, idx + len(word)


def find_pattern(pieces: List, whole: List) -> Tuple[int, int]:
    num_pieces = len(pieces)
    result = (0, 0)
    for i in (j for j, entry in enumerate(whole) if entry == pieces[0]):
        if whole[i : i + num_pieces] == pieces:
            result = (i, i + num_pieces)
    return result


def edit_distance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):

            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    return dp[m][n]


def argmin(lst: List) -> int:
    return min(range(len(lst)), key=lambda x: lst[x])


def argmax(lst: List) -> int:
    return max(range(len(lst)), key=lambda x: lst[x])


def find_index(
    context: str, word: str, method: Optional[str] = "regular"
) -> Tuple[int, int]:
    if method == "edit":
        tokenized = context.split()
        editdists = [edit_distance(w, word) for w in tokenized]

        index = argmin(editdists)
    else:
        # prefix, postfix = context.split(word)
        prefix, postfix = re.split(rf"\b{word}\b", context)
        word_length = len(word.split(" "))

        start = len(prefix.split())
        end = start + word_length

        # prefix = context.split(word)[0].strip().split()
        # index = len(prefix)

    return start, end


def gen_words(length: int) -> str:
    return " ".join([char for char in string.ascii_lowercase[0:length]])


def find_paired_indices(
    context: str, word1: str, word2: str, importance: int = 1
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if importance == 1:
        idx1 = find_index(context, word1)
        replace_cand = gen_words(len(word1.split()))
        idx2 = find_index(context.replace(word1, replace_cand), word2)
    else:
        idx2 = find_index(context, word2)
        replace_cand = gen_words(len(word2.split()))
        idx1 = find_index(context.replace(word2, replace_cand), word1)

    return idx1, idx2


def mask(sentence: str, word: str) -> str:
    replaced = re.sub(rf"(?<![\w\/-])({word})(?=[^\w\/-])", "[MASK]", sentence)
    masked = ["[CLS]"] + [replaced] + ["[SEP]"]
    return " ".join(masked)


def batch_wise_logprobs(logprobs, ids, reduction):
    batch_wise = [torch.stack(token_wise).T for token_wise in list(zip(*logprobs))]
    batch_labels = []
    for batch in batch_wise:
        batch_labels.append(torch.stack([reduction(y[i]) for i, y in zip(ids, batch)]))

    return torch.stack(batch_labels)


def leading_whitespace_behavior(tokenizer, n_random_words=1000):
    r = RandomWord()
    test_words = r.random_words(n_random_words)

    def is_sublist(superlist, sublist):
        """
        This checks if, e.g., [20, 764, 290] is a 'sublist'/'subsequence' of e.g. [152, 20, 764, 290].
        This is used to test the following: if "<word>" is tokenized as [tokenID_1], is " <word>" tokenized as [whitespace_ID, tokenID_1], or as [tokenID_2]?
        i.e. how does the tokenizer treat leading whitespace?
        """
        sublist_bool = any(
            superlist[idx : idx + len(sublist)] == sublist
            for idx in range(len(superlist) - len(sublist) + 1)
        )
        return sublist_bool

    divergences = (
        []
    )  # How many words are tokenized differently when fed in with a leading whitespace?
    for word in test_words:
        no_leading_space = tokenizer.encode(word, add_special_tokens=False)
        leading_space = tokenizer.encode(f" {word}", add_special_tokens=False)
        divergences.append(is_sublist(leading_space, no_leading_space))
    #
    average = sum(divergences) / len(divergences)
    if (
        average <= 0.05
    ):  # Classify leading whitespace behavior on this kind of graded spectrum
        return "gpt2"  # tokenizers like GPT2's generally encode "word" as [tokenID_1] and " word" as [tokenID_2]. The encoding for "something something word something" is likely to include tokenID_2, not tokenID_1.
    elif 0.05 < average <= 0.5:
        return "gpt2-mixed"  # Generally like GPT2 tokenizer behavior, but not 100%
    elif 0.5 < average <= 0.95:
        return "llama-mixed"  # Generally like Llama tokenizer behavior, but not 100%
    else:
        return "llama"  # tokenizers like the Llama Tokenizer seem to generally encode "word" as [token_ID1] and " word" as [whitespace_ID, tokenID_1]. The encoding "something something word something" is likely to include tokenID_1.
