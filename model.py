import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, label_size):
        super(LSTM, self).__init__()
        # init all class attribute
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # custom all layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight.requires_grad = False
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=2, bidirectional=True, dropout=0.5,
                           batch_first=True)
        self.fc = nn.Linear(self.hidden_dim * 2, label_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        # implement the forward function of the model.
        embedding = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedding)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out
