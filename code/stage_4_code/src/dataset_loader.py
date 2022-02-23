'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from .base.dataset import dataset

import os
import random
import string
import pandas as pd
import torch
from torchtext.legacy.data import Field, LabelField, TabularDataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class Dataset_Loader(dataset):

    text_field = None
    label_field = None
    train_data = None
    test_data = None
    
    def __init__(self, dName=None, dDescription=None, dPath=None):
        super().__init__(dName, dDescription, dPath)

    def __len__(self):
        return len(self.data['X'])

    def txt2csv(self):
        print('Converting txt files to train.csv and test.csv.')
        print('Doing preprocessing...')
        steps = ['train', 'test']
        porter = PorterStemmer()
        table = str.maketrans('', '', string.punctuation)
        stop_words = set(stopwords.words('english'))

        for step in steps:
            print('Making ' + step + '.csv...')
            pos = os.listdir(self.dataset_source_file_path + step + '\\pos\\')
            pos = [step + '\\pos\\' + name for name in pos]
            neg = os.listdir(self.dataset_source_file_path + step + '\\neg\\')
            neg = [step + '\\neg\\' + name for name in neg]
            all_data = pos + neg
            random.Random(48).shuffle(all_data)

            X = []
            y = []
            for file in all_data:
                if file.endswith(".txt"):
                    try:
                        f = open(self.dataset_source_file_path + file, 'r', encoding='utf-8')

                        # Load the raw text.
                        # Split into tokens.
                        # Convert to lowercase.
                        # Remove punctuation from each token.
                        # Filter out remaining tokens that are not alphabetic.
                        # Filter out tokens that are stop words.
                        # Stem the remaining words.
                        # https://machinelearningmastery.com/clean-text-machine-learning-python/
                        tokens = word_tokenize(f.read())
                        stripped = [w.translate(table) for w in tokens]
                        words = [word for word in stripped if word.isalpha()]
                        words = [porter.stem(w) for w in words if not w in stop_words]
                        text = ' '.join(words)
                        label = 1 if file.split('\\')[1] == 'pos' else 0

                        if len(text) <= 1:
                            continue

                        X.append(text)
                        y.append(label)

                    except FileNotFoundError:
                        print("Dataset file not found")
                        exit(1)

            df = pd.DataFrame(data={'text': X, 'label': y})
            df.to_csv(self.dataset_source_file_path + step + '.csv', index=False)

    def load(self, classification=True):
        print('Loading training and testing data...')

        # For classification task
        if classification:

            self.text_field = Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True,
                                    batch_first=True)
            self.label_field = LabelField(sequential=False, use_vocab=False, dtype=torch.float, batch_first=True)

            fields = [('text', self.text_field), ('label', self.label_field)]

            self.train_data, self.test_data = TabularDataset.splits(path=self.dataset_source_file_path,
                                                                    train='train.csv',
                                                                    test='test.csv',
                                                                    format='CSV',
                                                                    fields=fields,
                                                                    skip_header=True)

            print('Loading training and testing data completes!')

            # Build vocab
            self.text_field.build_vocab(self.train_data, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)

        else:
            print('TBD')

