import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import csv
import os
import torch
from itertools import zip_longest
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    test_id = [ test_id['id'] for test_id in data]
    # TODO: crecate DataLoader for test dataset

    test_dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len , True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = args.batch_size,
                                            shuffle = False,
                                            num_workers = 2)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = torch.load(args.ckpt_path)
    model.eval()
    predict_output = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            predict = torch.argmax(outputs , dim = 1)
            predict_output += predict.int().tolist()

    predict_label = [test_dataset.idx2label[predict] for predict in predict_output]
    
    # write predict csv
    print("save csv ...")
    d = [test_id, predict_label]
    export_data = zip_longest(*d, fillvalue = '')
    with open(args.pred_file, 'w' , encoding="utf-8", newline='') as fp:
      wr = csv.writer(fp)
      wr.writerow(("id", "intent"))
      wr.writerows(export_data)
    fp.close()
    print("Finish Predicting")
    # load weights into model

    # TODO: predict dataset

    # TODO: write prediction to file (args.pred_file)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/ckpt3.model",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
