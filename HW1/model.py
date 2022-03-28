from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding
from torchvision.models import resnet18, resnet101
import torch.nn.functional as F

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        pad_idx
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embeddings = Embedding.from_pretrained(embeddings, freeze=False, padding_idx= pad_idx)
        self.embedding_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = num_class
        self.rnn = nn.LSTM(embeddings.size(1), hidden_size, num_layers = num_layers,bidirectional = bidirectional,
        batch_first= True)
        cnn = []
        for i in range(2):
            conv_layer = nn.Sequential(
                nn.Conv1d(self.embedding_dim, self.embedding_dim, 5, 1, 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            cnn.append(conv_layer)
        self.cnn = nn.ModuleList(cnn)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.classifier = nn.Sequential( 
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.output_dim),
                                        )
        # TODO: model architecture

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch):
        # TODO: implement model forward
        inputs = self.embeddings(batch)
        # inputs = inputs.permute((0,2,1))
        # for conv in self.cnn:
        #     inputs = conv(inputs)
        # inputs = inputs.permute((0,2,1))
        x, _  = self.rnn(inputs,None)
        x = torch.mean(x, dim = 1)
        x = self.layer_norm(x)
        x = self.classifier(x)
        return x

class SlotClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        pad_idx
    ) -> None:
        super(SlotClassifier, self).__init__()
        self.embeddings = Embedding.from_pretrained(embeddings, freeze=False, padding_idx= pad_idx)
        self.embedding_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = num_class
        self.rnn = nn.GRU(embeddings.size(1), hidden_size, num_layers = num_layers,bidirectional = bidirectional,
        batch_first= True)
        cnn = []
        for i in range(2):
            conv_layer = nn.Sequential(
                nn.Conv1d(self.embedding_dim, self.embedding_dim, 5, 1, 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            cnn.append(conv_layer)
        self.cnn = nn.ModuleList(cnn)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.classifier = nn.Sequential( 
            # nn.Dropout(dropoudt),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.output_dim),
                                        )
        # TODO: model architecture

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch):
        # TODO: implement model forward
        inputs = self.embeddings(batch)
        # inputs = inputs.permute((0,2,1))
        # for conv in self.cnn:
        #     inputs = conv(inputs)
        # inputs = inputs.permute((0,2,1))
        x, _  = self.rnn(inputs,None)
        x = self.layer_norm(x)
        x = self.classifier(x)
        return x
