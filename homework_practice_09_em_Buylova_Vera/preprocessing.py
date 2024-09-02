from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    
    sentence_pairs = []
    alignments = []
    f = open(filename)
    tx = f.read()
    if '&' in tx:
        tx = tx.replace('&', '&#38;')
        tree = ET.fromstring(tx) 
    else:
        tree = ET.parse(filename)
    for sentence in tree.findall('s'):
        eng = sentence.find('english').text
        cz = sentence.find('czech').text
        possible_alignments = [
            tuple(map(int, val.split('-'))) for val in (sentence.find('possible').text or "").split()
        ]
        sure_alignments = [
            tuple(map(int, val.split('-'))) for val in (sentence.find('sure').text or "").split()
        ]
        alignments.append(LabeledAlignment(sure=sure_alignments, possible=possible_alignments))
        sentence_pairs.append(SentencePair(eng.split(), cz.split()))
    
    return sentence_pairs, alignments   





def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_freq = {}
    target_freq = {}
    for pair in sentence_pairs:
        for token in pair.source:
            source_freq[token] = source_freq.get(token, 0) + 1
        for token in pair.target:
            target_freq[token] = target_freq.get(token, 0) + 1
    if freq_cutoff is not None:
        sorted_source_tokens = sorted(source_freq, key=lambda x: source_freq[x], reverse=True)[:freq_cutoff]
        sorted_target_tokens = sorted(target_freq, key=lambda x: target_freq[x], reverse=True)[:freq_cutoff]
    else:
        sorted_source_tokens = sorted(source_freq)
        sorted_target_tokens = sorted(target_freq)
    source_dict = {token: index for index, token in enumerate(sorted_source_tokens)}
    target_dict = {token: index for index, token in enumerate(sorted_target_tokens)}

    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """

    tokenized_sentence_pairs = []
    for pair in sentence_pairs:
        source_tokens = [source_dict[token] for token in pair.source if token in source_dict]
        target_tokens = [target_dict[token] for token in pair.target if token in target_dict]
        if not source_tokens or not target_tokens:
            continue
        tokenized_sentence_pairs.append(TokenizedSentencePair(np.array(source_tokens, dtype=np.int32), np.array(target_tokens, dtype=np.int32)))

    return tokenized_sentence_pairs
