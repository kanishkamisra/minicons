import random
import re
import string
import torch
from polyleven import levenshtein

from typing import List, Tuple, Optional

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

def character_span(sentence, word):
    assert word in sentence
    idx = sentence.find(word)
    return idx, idx + len(word)

def find_pattern(pieces: List, whole: List) -> Tuple[int, int]:
    num_pieces = len(pieces)
    result = (0, 0)
    for i in (j for j,entry in enumerate(whole) if entry == pieces[0]):
        if whole[i:i+num_pieces] == pieces:
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
            elif word1[i-1] == word2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) 
  
    return dp[m][n]

def argmin(lst: List) -> int:
    return min(range(len(lst)), key=lambda x: lst[x])

def argmax(lst: List) -> int:
    return max(range(len(lst)), key=lambda x: lst[x])
  
def find_index(context: str, word: str, method: Optional[str] = "regular") -> Tuple[int, int]:
    if method == "edit":
        tokenized = context.split()
        editdists = [edit_distance(w, word) for w in tokenized]
        
        index = argmin(editdists)
    else:
        # prefix, postfix = context.split(word)
        prefix, postfix = re.split(fr"\b{word}\b", context)
        word_length = len(word.split(" "))

        start = len(prefix.split())
        end = start + word_length

        #prefix = context.split(word)[0].strip().split()
        #index = len(prefix)
    
    return start, end

def gen_words(length: int) -> str:
    return " ".join([char for char in string.ascii_lowercase[0:length]])

def find_paired_indices(context: str, word1: str, word2: str, importance: int = 1) -> Tuple[Tuple[int, int], Tuple[int, int]]:
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
    replaced = re.sub(rf'(?<![\w\/-])({word})(?=[^\w\/-])', '[MASK]', sentence)
    masked = ['[CLS]'] + [replaced] + ['[SEP]']
    return ' '.join(masked)