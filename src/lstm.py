#!/usr/bin/env python3

import gzip
import csv
import string
import numpy as np
import time
import abc
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

DATA_PATH = '../data/imdb-dataset.csv.gz'
POSITIVE_LABEL = 'positive'
NEGATIVE_LABEL = 'negative'
VALID_CHARS = string.ascii_lowercase + string.digits + ' '
INVALID_CHARS = set(string.printable).difference(VALID_CHARS)
LOWERCASE_TRANSLATOR = str.maketrans({c: '' for c in INVALID_CHARS})
NUM_EMBEDDING_DIMENSIONS = 100
NUM_HIDDEN_DIMENSIONS = 128
NUM_OUTPUT_DIMENSIONS = 1
BATCH_SIZE = 32
NUM_EPOCHS = 3


class SentimentClassifier:

    def __init__(self, train_data, test_data, vocab_size, padding_element=0):
        self.padding_element = padding_element
        self.train_dataset = MovieReviewsDataset(train_data)
        self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                       collate_fn=self.collate_fn)
        self.test_dataset = MovieReviewsDataset(test_data)
        self.test_loader = DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                      collate_fn=self.collate_fn)
        self.lstm = SentimentLSTM(vocab_size, NUM_EMBEDDING_DIMENSIONS, NUM_HIDDEN_DIMENSIONS, NUM_OUTPUT_DIMENSIONS)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.lstm.parameters())

    def collate_fn(self, batch):
        reviews, labels = zip(*batch)
        reviews_padded = pad_sequence(reviews, batch_first=True, padding_value=self.padding_element)
        labels = torch.tensor(labels, dtype=torch.float)
        return reviews_padded, labels

    def train_and_evaluate(self):
        for epoch in range(NUM_EPOCHS):
            print('Starting epoch ' + str(epoch + 1))
            start = time.perf_counter()
            self.train()
            avg_val_loss, avg_val_accuracy = self.evaluate()
            print(f"Epoch: {epoch + 1}, Validation loss: {avg_val_loss}, Validation accuracy: {avg_val_accuracy}",
                  str(time.perf_counter() - start) + 's')

    def train(self):
        self.lstm.train()
        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.lstm(inputs)
            loss = self.criterion(outputs.squeeze(), labels)
            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        self.lstm.eval()
        total_eval_loss = 0
        total_eval_accuracy = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.lstm(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                total_eval_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs.squeeze()))
                total_eval_accuracy += torch.sum(preds == labels).item()

        avg_val_loss = total_eval_loss / len(self.test_loader)
        avg_val_accuracy = total_eval_accuracy / len(self.test_dataset)
        return avg_val_loss, avg_val_accuracy


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(x)
        return self.output_layer(hidden[-1])


class MovieReviewsDataset(Dataset):
    def __init__(self, reviews):
        self.reviews = [torch.tensor(np.array(review[0]), dtype=torch.long) for review in reviews]
        self.labels = torch.tensor([review[1] for review in reviews], dtype=torch.float)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return self.reviews[idx], self.labels[idx]


class Space(abc.ABC):

    def __init__(self, labeled_reviews, padding_element=None):
        self.labeled_reviews = labeled_reviews
        self.training_reviews, self.testing_reviews = split_data(labeled_reviews)
        self.vectorized_training_reviews = None
        self.vectorized_testing_reviews = None
        self.padding = padding_element

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

    @property
    def size(self):
        raise NotImplementedError('Space size function not given!')

    @property
    def all_data(self):
        return self.training_data, self.testing_data

    @property
    def padding_element(self):
        if self.padding is None:
            raise NotImplementedError('No padding element given!')
        else:
            return self.padding


class KeyedWordSpace(Space):
    """
    Maps words to unique integer keys. The keys don't mean anything beyond that.
    """

    def __init__(self, labeled_reviews):
        super().__init__(labeled_reviews, padding_element=0)
        all_words = set(word for review in labeled_reviews for word in review[0])
        self.words_to_key = {word: i + 1 for i, word in enumerate(all_words)}

    def vectorized(self, review):
        return tuple(self.words_to_key[word] for word in review[0]), review[1]

    def size(self):
        return max(v for k, v in self.words_to_key.items())


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
        super().__init__(labeled_reviews, padding_element=SentimentSpace.PADDING_VECTOR)
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
        return tuple(self.sentiments_by_word[word] for word in labeled_review[0]), labeled_review[1]

    def size(self):
        return self.word_key


def main():
    start = time.perf_counter()
    space = KeyedWordSpace(preprocess_data(read_csv(yield_all_data_lines())))
    print('vectorized data', time.perf_counter() - start)
    train_data, test_data = space.all_data
    classifier = SentimentClassifier(train_data, test_data, space.size() + 1, space.padding_element)
    classifier.train_and_evaluate()


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
