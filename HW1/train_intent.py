import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import tensorflow as tf
import datetime
import torch
import torch.nn as nn
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier
from torch.utils.tensorboard import SummaryWriter

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

torch.manual_seed(878)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/intent/' + 'cnn-lstm' + '/train'
dev_log_dir = 'logs/intent/' + 'cnn-lstm' + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
dev_summary_writer = tf.summary.create_file_writer(dev_log_dir)

def evaluation(outputs, labels):
    predict = torch.argmax(outputs , dim = 1)
    correct = torch.sum(torch.eq(predict, labels)).item()
    return correct

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    # datasets: Dict[str, SeqClsDataset] = {
    #     split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
    #     for split, split_data in data.items()
    # }
    # TODO: crecate DataLoader for train / dev datasets\

    train_set = SeqClsDataset(data['train'], vocab, intent2idx, args.max_len, False)
    dev_set = SeqClsDataset(data['eval'], vocab, intent2idx, args.max_len, False)

    train_loader = torch.utils.data.DataLoader(dataset = train_set,
                                            batch_size = args.batch_size,
                                            shuffle = True,
                                            num_workers = 2)
    dev_loader = torch.utils.data.DataLoader(dataset = dev_set,
                                            batch_size = args.batch_size,
                                            shuffle = False,
                                            num_workers = 2)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        train_set.num_classes,
        vocab.pad_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    t_batch = len(train_loader) 
    v_batch = len(dev_loader)
    total_loss, total_acc, best_acc = 0, 0, 0
    batch_size = args.batch_size


    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        total_loss, total_acc = 0, 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad() 
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            correct = evaluation(outputs, labels) 
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct/batch_size*100), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', total_loss / t_batch, step=epoch)
            tf.summary.scalar('accuracy', total_acc / t_batch, step=epoch)
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(dev_loader):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(inputs)
                loss = criterion(outputs, labels) 
                correct = evaluation(outputs, labels) 
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(model, "{}/val_acc_{:.3f}.model".format(args.ckpt_dir,total_acc/v_batch*100))
                # torch.save(model, "{}/ckpt3.model".format(args.ckpt_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
            with dev_summary_writer.as_default():
                tf.summary.scalar('loss', total_loss / v_batch, step=epoch)
                tf.summary.scalar('accuracy', total_acc / v_batch, step=epoch)
        

        
        model.train()

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

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
