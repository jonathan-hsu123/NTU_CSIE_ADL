from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab
import torch


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        is_test: bool
    ):
        # self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self.idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.is_test = is_test
        self.data, self.label = self.collate_fn(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        if self.is_test == False:
            return self.data[index], self.label[index]
        return self.data[index]

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]):
        # TODO: implement collate_fn
        text = [data['text'].split() for data in samples]
        word2idx = self.vocab.encode_batch(batch_tokens = text, to_len = self.max_len)
        word2idx_tensor = torch.LongTensor(word2idx)
        if self.is_test == False :
            intents = [data['intent'] for data in samples] # get batch labels
            labels = [self.label2idx(intent) for intent in intents] # label to idx
            labels_tensor = torch.LongTensor(labels)
            return word2idx_tensor , labels_tensor
        return word2idx_tensor , None
        

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
