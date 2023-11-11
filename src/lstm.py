#!/usr/bin/env python3

import gzip
import csv
import string
import numpy as np
import time
import abc
import torch
from torch.utils.data import Dataset

DATA_PATH = '../data/imdb-dataset.csv.gz'
POSITIVE_LABEL = 'positive'
NEGATIVE_LABEL = 'negative'
VALID_CHARS = string.ascii_lowercase + string.digits + ' '
INVALID_CHARS = set(string.printable).difference(VALID_CHARS)
LOWERCASE_TRANSLATOR = str.maketrans({c: '' for c in INVALID_CHARS})


class MovieReviewsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.inputs = torch.tensor([item[0] for item in data], dtype=torch.float)
        self.labels = torch.tensor([item[1] for item in data], dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class Space(abc.ABC):

    def __init__(self, labeled_reviews):
        self.labeled_reviews = labeled_reviews
        self.training_reviews, self.testing_reviews = split_data(labeled_reviews)
        self.max_review_length = len(max(labeled_reviews, key=lambda r: len(r[0]))[0])
        self.vectorized_training_reviews = None
        self.vectorized_testing_reviews = None

    @property
    def training_data(self):
        if not self.vectorized_training_reviews:
            self.vectorized_training_reviews = tuple(self.vectorized(review) for review in self.training_reviews)
        return self.vectorized_training_reviews

    def vectorized(self, review):
        pass

    @property
    def testing_data(self):
        if not self.vectorized_testing_reviews:
            self.vectorized_testing_reviews = tuple(self.vectorized(review) for review in self.testing_reviews)
        return self.vectorized_testing_reviews

    def padded(self, review_vector, padding_vector):
        diff_to_max = self.max_review_length - len(review_vector)
        return review_vector + tuple(padding_vector for _ in range(diff_to_max))


class KeyedWordSpace(Space):
    """
    Maps words to unique integer keys. The keys don't mean anything beyond that.
    """

    def __init__(self, labeled_reviews):
        super().__init__(labeled_reviews)
        all_words = set(word for review in labeled_reviews for word in review[0])
        self.words_to_key = {word: i for i, word in enumerate(all_words)}

    def vectorized(self, review):
        return self.padded(tuple(self.words_to_key[word] for word in review[0]), -1), review[1]


class SentimentSpace(Space):
    """
    Maps words to sentiment vectors. Each sentiment vector has three elements
    in the following order:
    - a count of positive reviews the word is in
    - a count of negative reviews the word is in
    - a key identifying the word
    """

    PADDING_VECTOR = np.array([0, 0, 0], dtype=np.int)

    def __init__(self, labeled_reviews):
        super().__init__(labeled_reviews)
        self.sentiments_by_word = {}
        self.word_key = 1
        self.create_space()

    def create_space(self):
        for review in self.labeled_reviews:
            for word in review[0]:
                if word not in self.sentiments_by_word:
                    self.sentiments_by_word[word] = np.array([0, 0, self.word_key], dtype=np.int)
                    self.word_key += 1
                if review[1]:
                    self.sentiments_by_word[word][0] += 1
                else:
                    self.sentiments_by_word[word][1] += 1

    def vectorized(self, labeled_review):
        return self.padded(tuple(self.sentiments_by_word[word] for word in labeled_review[0]),
                           SentimentSpace.PADDING_VECTOR), labeled_review[1]


def main():
    start = time.perf_counter()
    all_alphanumeric_data = preprocess_data(read_csv(yield_all_data_lines()))
    print(time.perf_counter() - start)
    spaces = [SentimentSpace(all_alphanumeric_data), KeyedWordSpace(all_alphanumeric_data)]
    print(time.perf_counter() - start)
    for space in spaces:
        print(space.training_data[0])


def yield_all_data_lines():
    with gzip.open(DATA_PATH, mode='rt', encoding='utf-8') as file:
        for line in file:
            yield line


def read_csv(lines):
    reader = csv.reader(line for line in lines)
    return (row for row in reader)


def preprocess_data(data):
    processors = [
        verify_labels,
        labels_to_bools,
        to_lowercase,
        to_alphanumeric_words,
        to_word_tuples,
    ]
    return tuple(row for row in compose(processors, data))


def verify_labels(data):
    for i, d in enumerate(data):
        if i > 0:
            if d[1] != POSITIVE_LABEL and d[1] != NEGATIVE_LABEL:
                raise Exception(d[1])
            yield d


def labels_to_bools(data):
    for d in data:
        d[1] = 1 if d[1] == POSITIVE_LABEL else 0
        yield d


def to_lowercase(data):
    for d in data:
        d[0] = d[0].lower()
        yield d


def to_alphanumeric_words(data):
    for d in data:
        d[0] = d[0].translate(LOWERCASE_TRANSLATOR)
        yield d


def to_alphanum_chars(text):
    return ''.join([c for c in text if c in VALID_CHARS])


def to_word_tuples(data):
    for d in data:
        d[0] = tuple(d[0].split(' '))
        yield d


def compose(fns, data):
    if len(fns) == 1:
        return fns[0](data)
    return compose(fns[1:], fns[0](data))


def split_data(all_data):
    cutoff = len(all_data) // 2
    return [all_data[:cutoff], all_data[cutoff:]]


if __name__ == '__main__':
    main()
