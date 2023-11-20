#!/usr/bin/env python3

import gzip
import csv
import itertools
import string
import numpy as np
import time
import argparse
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
NUM_EMBEDDING_DIMENSIONS = 50
NUM_HIDDEN_DIMENSIONS = 128
NUM_OUTPUT_DIMENSIONS = 1
BATCH_SIZE = 32
NUM_EPOCHS = 50
USE_KEYED_FLAG = 'k'
USE_SENTIMENT_FLAG = 's'


class SentimentClassifier:

    def __init__(self, train_data, test_data, vocab_size, padding_element=0.0):
        self.padding_element = padding_element
        self.train_dataset = MovieReviewsDataset(train_data)
        self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                       collate_fn=self.collate_fn)
        self.test_dataset = MovieReviewsDataset(test_data)
        self.test_loader = DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                      collate_fn=self.collate_fn)
        self.lstm = SentimentLSTM(vocab_size, embedding_dim=NUM_EMBEDDING_DIMENSIONS, hidden_dim=NUM_HIDDEN_DIMENSIONS,
                                  output_dim=NUM_OUTPUT_DIMENSIONS)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.lstm.parameters())

    def collate_fn(self, batch):
        reviews, labels = zip(*batch)
        reviews_padded = pad_sequence(reviews, batch_first=True, padding_value=0.0)
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
        self.embedding = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embedding_dim, padding_idx=0)
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


def main():
    space_type, num_of_records = parse_args()
    start = time.perf_counter()
    train_data, test_data, size = vectorize(preprocess_data(read_csv(yield_all_data_lines(), num_of_records)),
                                            choose_space(space_type))
    print('vectorized data', time.perf_counter() - start)
    classifier = SentimentClassifier(train_data, test_data, size)
    classifier.train_and_evaluate()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--space', choices=[USE_KEYED_FLAG, USE_SENTIMENT_FLAG], default=USE_KEYED_FLAG)
    parser.add_argument('-n', '--num_of_records', type=int, default=-1,
                        help="The number of records to read from file. Defaults to all.")
    args = parser.parse_args()
    return args.space, args.num_of_records


def yield_all_data_lines():
    with gzip.open(DATA_PATH, mode='rt', encoding='utf-8') as file:
        for line in file:
            yield line


def read_csv(lines, num_of_records):
    reader = csv.reader(line for line in lines)
    if num_of_records > 0:
        return itertools.islice((row for row in reader), num_of_records)
    else:
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


def choose_space(space_type):
    if space_type is USE_SENTIMENT_FLAG:
        print('Using sentiment-based space')
        return to_sentiment_score_vectors
    else:
        print('Using key-based space')
        return to_key_number_vectors


def to_sentiment_score_vectors(labeled_reviews):
    counts_by_word = count_word_frequencies(labeled_reviews)
    words_to_key = {word: (counts[0] - counts[1] / (counts[0] + counts[1])) for word, counts in counts_by_word.items()}
    vectorized_data = tuple((tuple(words_to_key[word] for word in review[0]), review[1]) for review in labeled_reviews)
    training_data, test_data = split_data(vectorized_data)
    return training_data, test_data, len(counts_by_word)


def count_word_frequencies(labeled_reviews):
    result = {}
    for review in labeled_reviews:
        for word in review[0]:
            if word not in result:
                result[word] = np.array([0, 0], dtype=np.int32)
            if review[1]:
                result[word][0] += 1
            else:
                result[word][1] += 1
    return result


def to_key_number_vectors(labeled_reviews):
    all_words = set(word for review in labeled_reviews for word in review[0])
    words_to_key = {word: i + 1 for i, word in enumerate(all_words)}
    vectorized_data = tuple((tuple(words_to_key[word] for word in review[0]), review[1]) for review in labeled_reviews)
    training_data, test_data = split_data(vectorized_data)
    return training_data, test_data, len(all_words)


def vectorize(labeled_reviews, to_vectors):
    return to_vectors(labeled_reviews)


def split_data(all_data):
    cutoff = len(all_data) // 2
    return [all_data[:cutoff], all_data[cutoff:]]


if __name__ == '__main__':
    main()
