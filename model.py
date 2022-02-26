from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding


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
        x, _  = self.rnn(inputs,None)
        x = torch.mean(x, dim = 1)
        x = self.layer_norm(x)
        x = self.classifier(x)
        return x
