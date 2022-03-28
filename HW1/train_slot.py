import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from numpy import corrcoef

import torch
import torch.nn as nn
from tqdm import trange

import tensorflow as tf
import datetime

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

from dataset import SlotClsDataset
from utils import Vocab
from model import SlotClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

torch.manual_seed(878)

train_log_dir = 'logs/slot/' + 'cnn-lstm' + '/train'
dev_log_dir = 'logs/slot/' + 'cnn-lstm' + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
dev_summary_writer = tf.summary.create_file_writer(dev_log_dir)


def evaluation(outputs, labels, pad_idx):
    predict = torch.argmax(outputs , dim = 1)
    non_pad = (labels != pad_idx)
    correct = torch.sum(torch.eq(predict[non_pad], labels[non_pad])).item()
    return correct / torch.sum(non_pad == True)

def gen_joint(outputs , labels, predict_output , truth_labels , dev_set):
    predict = torch.argmax(outputs , dim = 2).int().tolist()
    labels = labels.view(-1,args.max_len).int().tolist()
    for data in predict:
        tags_label = [dev_set.idx2label[tag] for tag in data if tag != dev_set.label2idx("Pad")]
        predict_output.append(tags_label)
    
    for label in labels:
        tags_label = [dev_set.idx2label[tag] for tag in label if tag != dev_set.label2idx("Pad")]
        truth_labels.append(tags_label)
    
    return predict_output, truth_labels
    

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    train_set = SlotClsDataset(data['train'], vocab, tag2idx, args.max_len, False)
    dev_set = SlotClsDataset(data['eval'], vocab, tag2idx, args.max_len, False)

    train_loader = torch.utils.data.DataLoader(dataset = train_set,
                                            batch_size = args.batch_size,
                                            shuffle = True,
                                            num_workers = 2)
    dev_loader = torch.utils.data.DataLoader(dataset = dev_set,
                                            batch_size = args.batch_size,
                                            shuffle = False,
                                            num_workers = 2)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SlotClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        train_set.num_classes,
        vocab.pad_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    t_batch = len(train_loader) 
    v_batch = len(dev_loader)
    total_loss, total_acc, best_acc = 0, 0, 0
    batch_size = args.batch_size

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        total_loss, total_acc = 0, 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad() 
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)
            correct = evaluation(outputs, labels, train_set.label2idx('Pad'))
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step() 
            total_acc += correct
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100), end='\r')
        # print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', total_loss / t_batch, step=epoch)
            tf.summary.scalar('token accuracy', total_acc.cpu() / t_batch, step=epoch)
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(dev_loader):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(inputs)
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)
                correct = evaluation(outputs, labels, train_set.label2idx('Pad'))
                loss = criterion(outputs, labels) 
                y_pred, y_true = gen_joint(outputs.view(-1,args.max_len,outputs.shape[-1]), labels, y_pred , y_true , dev_set)
                total_acc += correct
                total_loss += loss.item()


            # print("Valid | Loss:{:.5f} Token Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            joint_acc = 0
            for i in range(len(y_true)):
                if str(y_pred[i]) == str(y_true[i]):
                    joint_acc += 1
            # print("Valid | Joint Acc : {:.3f}%".format(joint_acc / len(y_true) * 100))
            # print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))
            # _f1_score = f1_score(y_true, y_pred) 
            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(model, "{}/Tok_Acc_{:.3f}.model".format(args.ckpt_dir, total_acc/v_batch*100))
                # torch.save(model, "{}/ckpt.model".format(args.ckpt_dir))
                # print('saving model with acc {:.3f}'.format(joint_acc / len(y_true) * 100))
            with dev_summary_writer.as_default():
                tf.summary.scalar('loss', total_loss / v_batch, step=epoch)
                tf.summary.scalar('token accuracy', total_acc.cpu() / v_batch, step=epoch)
                tf.summary.scalar('joint accuracy', joint_acc / len(y_true), step=epoch)
                # tf.summary.scalar('f1 score', _f1_score, step=epoch)
        model.train()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=6e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)