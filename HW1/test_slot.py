import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import csv
import os
import torch
from itertools import zip_longest
from dataset import SlotClsDataset
from model import SlotClassifier
from utils import Vocab

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    test_id = [ test_id['id'] for test_id in data]

    test_set = SlotClsDataset(data, vocab, tag2idx, args.max_len, True)

    test_loader = torch.utils.data.DataLoader(dataset = test_set,
                                            batch_size = args.batch_size,
                                            shuffle = False,
                                            num_workers = 2)
    model = torch.load(args.ckpt_path)
    model.eval()
    predict_output = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs.view(-1,args.max_len,outputs.shape[-1])
            predict = torch.argmax(outputs , dim = 2)
            predict = predict.int().tolist()
            for batch_data in predict:
                tags_label = [test_set.idx2label[int(tag)] for tag in batch_data if tag != test_set.label2idx('Pad')]
                tags_predict = " ".join(elem for elem in tags_label)
                predict_output.append(tags_predict)


    # write predict csv
    print("save csv ...")
    d = [test_id, predict_output]
    export_data = zip_longest(*d, fillvalue = '')
    with open(args.pred_file, 'w' , encoding="utf-8", newline='') as fp:
      wr = csv.writer(fp)
      wr.writerow(("id", "tags"))
      wr.writerows(export_data)
    fp.close()
    print("Finish Predicting")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/ckpt3.model",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)