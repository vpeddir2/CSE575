#!/usr/bin/env python3

import gzip
import csv
import itertools
import string
import numpy as np
import pandas as pd
import time
import argparse
import re
import torch
import random
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords, wordnet
# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


DATA_PATH = '../data/imdb-dataset.csv.gz'

class Preprocessor:

    POSITIVE_LABEL = 'positive'
    NEGATIVE_LABEL = 'negative'
    VALID_CHARS = string.ascii_lowercase + string.digits + ' '
    INVALID_CHARS = set(string.printable).difference(VALID_CHARS)
    LOWERCASE_TRANSLATOR = str.maketrans({c: '' for c in INVALID_CHARS})

    def __init__(self):
        pass
        # self.lm = WordNetLemmatizer()
        # self.pos_dict = {'A': wordnet.ADJ, 'N': wordnet.NOUN, 'R': wordnet.ADV, 'V': wordnet.VERB}
        # self.stop_words = stopwords.words('english')
        # self.stop_words.remove('not')

    def preprocess_data(self, data):
        processors = [
            # self.verify_labels,
            self.labels_to_bools,
            self.expand_contractions,
            self.to_lowercase,
            self.remove_html_tags,
            self.remove_urls,
            # self.remove_stopwords_and_lemmatize,
            self.remove_all_special_chars,
            self.to_tuples
        ]
        return tuple(row for row in self.compose(processors, data))


    def labels_to_bools(self, data):
        for d in data:
            d[1] = 1 if d[1] == Preprocessor.POSITIVE_LABEL else 0
            yield d


    def to_lowercase(self, data):
        for d in data:
            d[0] = d[0].lower()
            yield d

    def remove_html_tags(self, data):
        for d in data:
            d[0] = d[0].replace('<br /><br />', ' ').replace('\x85', ' ').replace('\x96', ' ')
            yield d

    def expand_contractions(self, data):
        for d in data:
            d[0] = d[0].replace('n\'t', ' not').replace('\'ve', ' have').replace('\'ll', ' will').replace('\'em', ' them')
            yield d
            
    def remove_urls(self, data):
        for d in data:
            pattern = re.compile(r'https?://\S+|www\.\S+')
            d[0]=pattern.sub(r'', d[0])
            yield d

    # def remove_stopwords_and_lemmatize(self, data):
    #     for d in tqdm(data):
    #         updated_pos = []
    #         sentence = ''
    #         pos_text = pos_tag(word_tokenize(d[0]))

    #         for word, tag in pos_text:
    #             if word.lower() not in self.stop_words:
    #                 updated_pos.append((word, self.pos_dict.get(tag[0].upper(), wordnet.NOUN)))

    #         for word, tag in updated_pos:
    #             if not tag:
    #                 sentence += ' ' + word
    #             else:
    #                 sentence += ' ' + self.lm.lemmatize(word, tag)

    #         d[0] = sentence.strip()
    #         yield d
            
    def remove_all_special_chars(self, data):
        for d in data:
            d[0] = re.sub(r'\s[a-z]\s', ' ', re.sub('[^a-z\s]', '', d[0]))
            yield d

    def to_tuples(self, data):
        for d in data:
            d[0] = tuple(d[0].split(' '))
            yield d

    def compose(self, fns, data):
        if len(fns) == 1:
            return fns[0](data)
        return self.compose(fns[1:], fns[0](data))

class LSTMVectorizer:
    
    SIMPLE_KEY_VECTORIZER = 'simple-key'
    ORDERED_SENTIMENT_VECTORIZER = 'ordered-sentiment'

    def __init__(self, data):
        self.data = data

    def vectorize(self, space_type):
        to_vectors = self.choose_vectorizer(space_type)
        return to_vectors(self.data)


    def choose_vectorizer(self, space_type):
        if space_type == LSTMVectorizer.ORDERED_SENTIMENT_VECTORIZER:
            return self.to_ordered_sentiment_indexes
        else:
            return self.to_key_number_vectors


    def to_ordered_sentiment_indexes(self, labeled_reviews):
        counts_by_word = self.count_word_frequencies(labeled_reviews)
        words_to_key = self.to_keys_by_word(counts_by_word)
        vectorized_data = tuple((tuple(words_to_key[word] for word in review[0]), review[1]) for review in labeled_reviews)
        training_data, test_data = self.split_data(vectorized_data)
        return training_data, test_data, len(words_to_key)


    def count_word_frequencies(self, labeled_reviews):
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

    def to_keys_by_word(counts_by_word):
        scores_by_word = {}
        for word in counts_by_word:
            counts = counts_by_word[word]
            scores_by_word[word] = counts[0]-counts[1] / (counts[0]+counts[1])
        scores_ordered = OrderedDict(sorted(scores_by_word.items(), key=lambda x: x[1]))
        return {word: i+1 for i, word in enumerate(scores_ordered)}

    def to_key_number_vectors(self, labeled_reviews):
        all_words = set(word for review in labeled_reviews for word in review[0])
        words_to_key = {word: i + 1 for i, word in enumerate(all_words)}
        vectorized_data = tuple((tuple(words_to_key[word] for word in review[0]), review[1]) for review in labeled_reviews)
        training_data, test_data = self.split_data(vectorized_data)
        return training_data, test_data, len(all_words)

    def split_data(self, all_data):
        cutoff = len(all_data) // 2
        data = random.sample(all_data, len(all_data))
        return [data[:cutoff], data[cutoff:]]


class LSTMSentimentClassifier:

    NUM_EMBEDDING_DIMENSIONS = 50
    NUM_HIDDEN_DIMENSIONS = 128
    NUM_OUTPUT_DIMENSIONS = 1
    BATCH_SIZE = 32
    NUM_EPOCHS = 50

    def __init__(self, train_data, test_data, vocab_size, padding_element=0.0):
        self.padding_element = padding_element
        self.train_dataset = MovieReviewsDataset(train_data)
        self.train_loader = DataLoader(self.train_dataset, batch_size=LSTMSentimentClassifier.BATCH_SIZE, shuffle=True,
                                       collate_fn=self.collate_fn)
        self.test_dataset = MovieReviewsDataset(test_data)
        self.test_loader = DataLoader(self.test_dataset, batch_size=LSTMSentimentClassifier.BATCH_SIZE, shuffle=False,
                                      collate_fn=self.collate_fn)
        self.lstm = SentimentLSTM(vocab_size, embedding_dim=LSTMSentimentClassifier.NUM_EMBEDDING_DIMENSIONS, hidden_dim=LSTMSentimentClassifier.NUM_HIDDEN_DIMENSIONS,
                                  output_dim=LSTMSentimentClassifier.NUM_OUTPUT_DIMENSIONS)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.lstm.parameters())

    def collate_fn(self, batch):
        reviews, labels = zip(*batch)
        reviews_padded = pad_sequence(reviews, batch_first=True, padding_value=0.0)
        labels = torch.tensor(labels, dtype=torch.float)
        return reviews_padded, labels

    def train_and_evaluate(self):
        for epoch in range(LSTMSentimentClassifier.NUM_EPOCHS):
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
    preprocessor = Preprocessor()
    classifier = LSTMVectorizer(data=preprocessor.preprocess_data(read_csv(yield_all_data_lines(), num_of_records)))
    train_data, test_data, size = classifier.vectorize(space_type)
    classifier = LSTMSentimentClassifier(train_data, test_data, size)
    classifier.train_and_evaluate()
    print(time.perf_counter() - start)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vectorizer', choices=[LSTMVectorizer.SIMPLE_KEY_VECTORIZER, LSTMVectorizer.ORDERED_SENTIMENT_VECTORIZER], default=LSTMVectorizer.SIMPLE_KEY_VECTORIZER)
    parser.add_argument('-n', '--num_of_records', type=int, default=-1,
                        help="The number of records to read from file. Defaults to all.")
    args = parser.parse_args()
    return args.vectorizer, args.num_of_records

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

if __name__ == '__main__':
    main()
