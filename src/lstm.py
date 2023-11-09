#!/usr/bin/env python3

import gzip
import csv
import string
import array
import time

DATA_PATH = '../data/imdb-dataset.csv.gz'
POSITIVE_LABEL = 'positive'
NEGATIVE_LABEL = 'negative'
VALID_CHARS = string.ascii_lowercase + string.digits + ' '
INVALID_CHARS = set(string.printable).difference(VALID_CHARS)
LOWERCASE_TRANSLATOR = str.maketrans({c: '' for c in INVALID_CHARS})


class SentimentSpace:
    """
    Maps words to sentiment vectors. Each sentiment vector has three elements
    in the following order:
    - a count of positive reviews the word is in
    - a count of negative reviews the word is in
    - a key identifying the word
    """

    PADDING_VECTOR = array.array('I', [0, 0, 0])

    def __init__(self, labeled_reviews):
        self.sentiments_by_word = {}
        self.labeled_reviews = labeled_reviews
        self.word_key = 1
        self.vectorized_reviews = None
        self.create_space()

    def create_space(self):
        for review in self.labeled_reviews:
            for word in review[0]:
                if word not in self.sentiments_by_word:
                    self.sentiments_by_word[word] = array.array('I', [0, 0, self.word_key])
                    self.word_key += 1
                if review[1]:
                    self.sentiments_by_word[word][0] += 1
                else:
                    self.sentiments_by_word[word][1] += 1

    @property
    def space(self):
        return tuple(self.sentiments_by_word.values())

    @property
    def reviews(self):
        if not self.vectorized_reviews:
            self.vectorized_reviews = tuple(self.vectorized(review) for review in self.labeled_reviews)
        return self.vectorized_reviews

    def vectorized(self, labeled_review):
        return tuple(self.sentiments_by_word[word] for word in labeled_review[0]), labeled_review[1]


def main():
    start = time.perf_counter()
    all_alphanumeric_data = preprocess_data(read_csv(yield_all_data_lines()))
    training_data, testing_data = split_data(all_alphanumeric_data)
    print(time.perf_counter() - start)

    sentiment_space = SentimentSpace(training_data)
    print(time.perf_counter() - start)
    print(len(sentiment_space.reviews))


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