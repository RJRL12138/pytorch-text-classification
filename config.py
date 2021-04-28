from torchtext import data

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import random


class Config:
    def __init__(self, filename):
        print('Configuration start')
        STOP_WORDS.add('n')
        self.train_data, self.val_data, self.dataset = None, None, None
        self.train_iter, self.val_iter = None, None
        self.test_data, self.test_iter = None, None
        self.TEXT = data.Field(sequential=True, tokenize=self.tokenizer, lower=True,
                               include_lengths=True, batch_first=True, fix_length=200,
                               stop_words=STOP_WORDS)
        self.LABEL = data.LabelField()
        self.spacy_en = spacy.load('en_core_web_sm')
        self.vocab_size, self.label_size = 0, 0
        self.load(filename)
        print("configuration finished")

    embedding_dim = 50
    hidden_dim = 128
    batch_size = 128
    lr = 1e-3
    EPOCHS = 30

    def print_para(self):
        print('embedding_dim:{}\nhidden_dim:{}\nbatch_size:{}\nlearning_rate:{}\nepochs:{}\n'.format(
            self.embedding_dim, self.hidden_dim, self.batch_size, self.lr, self.EPOCHS))

    def tokenizer(self, text):  # create a tokenizer function
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def load(self, filename):
        self.dataset = data.TabularDataset(path=filename[0], format='csv', skip_header=True,
                                           fields=[('label', self.LABEL), ('text', self.TEXT)])

        self.train_data, self.val_data = self.dataset.split(random_state=random.seed(42))
        self.train_data, self.test_data = self.train_data.split(random_state=random.seed(42))
        vector_str = 'glove.6B.{}d'.format(self.embedding_dim)
        self.TEXT.build_vocab(self.train_data, max_size=10000, vectors=vector_str)
        self.LABEL.build_vocab(self.train_data)

        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (self.train_data, self.val_data, self.test_data), sort=False, device="cuda", repeat=False, shuffle=True,
            batch_size=self.batch_size)
        self.vocab_size = len(self.TEXT.vocab)
        self.label_size = len(self.LABEL.vocab)


if __name__ == '__main__':
    cfg = Config(['./data/clean.csv'])
    for batch in cfg.test_iter:
        print(batch.text[0])
