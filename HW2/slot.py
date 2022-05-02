from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
import json 
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List
from datasets import Dataset
import numpy as np


def compute_metrics(pred):
    logits, labels = pred 
    prediction = np.argmax(logits, axis=-1)
    correct_num = 0

    for p, l in zip(prediction, labels):
        if (p == l).all():
            correct_num += 1
    return {'accuracy': correct_num / len(prediction)}

def open_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)

def main(args):
    tag2idx: Dict[str, int] = json.loads(args.tag_to_idx_data_path.read_text())
    tag2idx.update({"[PAD]":9})
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    train_json = open_json(args.train_data_path)
    valid_json = open_json(args.valid_data_path)
    test_json = open_json(args.test_data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    def map_data(input_data) -> Dict[str, List]:
        return_data = {}
        for i, instance in enumerate(input_data):
            text = instance['tokens']
            while len(text) < 128:
                text.append("[PAD]")
            input_ids = tokenizer.convert_tokens_to_ids(text)
            token_type_ids = [0] * len(input_ids)
            attention_mask = [1] * len(input_ids)
            tokens = {'input_ids':input_ids, 'token_type_ids':token_type_ids, "attention_mask":attention_mask}
            if i == 0:
                return_data.update({key: [] for key in tokens.keys()})
                if 'tags' in instance:
                    return_data.update({'labels': []})
            for key, values in tokens.items():
                return_data[key].append(values)
            if 'tags' in instance:
                return_data['labels'].append([tag2idx[instance['tags'][i]] if i < len(instance['tags']) else 9 for i in range(128)])
        return return_data

    train_dataset = Dataset.from_dict(map_data(train_json))
    valid_dataset = Dataset.from_dict(map_data(valid_json))
    test_dataset = Dataset.from_dict(map_data(test_json))

    seed = 7789

    training_args = TrainingArguments(
        output_dir=args.model_dir, # output directory where model predictions and checkpoint will be saved 
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=128,
        logging_dir=args.log_dir,
        logging_strategy='steps',
        logging_steps=50,
        seed=seed,
        save_strategy='epoch',
        evaluation_strategy='steps',
        eval_steps=50,
        learning_rate=args.lr,
        fp16=True,
        num_train_epochs=args.num_epoch,
        gradient_checkpointing=True,
    )
    config = AutoConfig.from_pretrained(args.model_type)
    model = AutoModelForTokenClassification.from_pretrained(args.model_type, num_labels=10)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

    training_args = TrainingArguments(
        output_dir='tmp2',
        do_train=False,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args
    )

    logits = trainer.predict(test_dataset)

    predictions = np.argmax(logits[0], axis=-1)
    pred_tags = []

    for i, instance in enumerate(test_json):
        test_id = instance['id']
        tags_id = predictions[i]
        tail = 0
        for j in range(len(tags_id) - 1, -1, -1):
            if tags_id[j] != 9:
                tail = j 
                break
        pred_tag = ''
        for j in range(tail + 1):
            pred_tag += idx2tag[tags_id[j]]
            if j != tail:
                pred_tag += ' '      
        pred_tags.append((test_id, pred_tag))
    
    with open(args.pred_file, 'w+') as f:
        f.write('id,intent\n')
        for i, e in pred_tags:
            f.write(f'{i},{e}\n')




def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        type=Path,
        help="Path to the data(input question).",
        default="/tmp2/jonathan/data/slot/train.json",
    )
    parser.add_argument(
        "--valid_data_path",
        type=Path,
        help="Path to the data(input question).",
        default="/tmp2/jonathan/data/slot/eval.json",
    )
    parser.add_argument(
        "--test_data_path",
        type=Path,
        help="Path to the data(input question).",
        default="/tmp2/jonathan/data/slot/test.json",
    )
    parser.add_argument(
        "--tag_to_idx_data_path",
        type=Path,
        help="Path to the data(input question).",
        default="/tmp2/jonathan/cache/slot/tag2idx.json",
    )
    
    # data
    parser.add_argument("--max_len", type=int, default=512)
    # optimizer
    parser.add_argument("--lr", type=float, default=5e-5)
    # data loader
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epoch", type=int, default=20)
    # bert model
    parser.add_argument("--model_type", type=str, default="bert-base-cased")
    parser.add_argument("--model_dir", type=Path, default="/tmp2/jonathanhsu/ADL/bert_base/slot/model_dir")
    parser.add_argument("--log_dir", type=Path, default="/tmp2/jonathanhsu/ADL/bert_base/slot/log_dir")

    parser.add_argument(
        "--pred_file",
        type=Path,
        help="File path to the prediction file",
        default="./pred_slot.csv"
    )
    
    args = parser.parse_args()
    return args 



if __name__ == "__main__":
    args = parse_args()
    main(args)