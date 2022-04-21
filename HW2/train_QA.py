from transformers import (
    AutoModelForQuestionAnswering,
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
    answers, labels = pred
    preds = np.argmax(answers, axis=-1)

    correct_numbers = 0
    for i in range(len(labels[0])):
        if preds[0][i] == labels[0][i] and preds[1][i] == labels[1][i]:
            correct_numbers += 1

    return {'accuracy': correct_numbers / len(labels[0])}

def open_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)

def main(args):
    train_json = open_json(args.train_data_path)
    valid_json = open_json(args.valid_data_path)
    context = open_json(args.context_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    def json_to_dict(json_file, context):
        return_dict = {'start_positions': [], 'end_positions': []}
        tokens_range = []
        total_len = 0
        for i, instance in enumerate(json_file):
            paragraph = context[instance['relevant']]
            question = instance['question']
            tokens = tokenizer(question, paragraph, padding='max_length', max_length=512, truncation='only_second', return_offsets_mapping=True, return_overflowing_tokens=True, stride=128)
            if i == 0:
                return_dict.update({key: [] for key in tokens.keys()})
            tmp = []
            for id in range(len(tokens['input_ids'])):
                tmp.append({k : v[id] for k, v in tokens.items()})
            tokens = tmp
            start = instance['answer']['start']
            end = start + len(instance['answer']['text'])
            start_pos, end_pos, token_id = -1, -1, 0
            tokens_range.append((total_len, total_len + len(tokens)))
            total_len += len(tokens)
            for token_index, token in enumerate(tokens):
                offsets = token['offset_mapping']
                start_p, end_p = -1, -1
                cnt = 0
                for offset_id, offset in enumerate(offsets):
                    if offset[0] == 0 and offset[1] == 0:
                        cnt += 1
                    if offset[0] <= start and offset[1] > start and cnt >= 2:
                        start_p = offset_id
                    if offset[0] < end and offset[1] >= end and cnt >= 2:
                        end_p = offset_id
                if start_p > -1 and end_p > -1:
                    token_id = token_index
                    start_pos, end_pos = start_p, end_p
            for token_index, token in enumerate(tokens):
                for k, v in token.items():
                    return_dict[k].append(v)
                if token_index == token_id:
                    return_dict['start_positions'].append(start_pos)
                    return_dict['end_positions'].append(end_pos)
                else:
                    return_dict['start_positions'].append(0)
                    return_dict['end_positions'].append(0)
        return return_dict, tokens_range
    train_dataset_dict, train_tokens_range = json_to_dict(train_json, context)            
    valid_dataset_dict, valid_tokens_range = json_to_dict(valid_json, context) 

    train_dataset = Dataset.from_dict(train_dataset_dict)
    valid_dataset = Dataset.from_dict(valid_dataset_dict)          

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
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_type)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--context_path",
        type=Path,
        help="Path to the context file.",
        default="./context.json",
    )
    parser.add_argument(
        "--train_data_path",
        type=Path,
        help="Path to the data(input question).",
        default="./train.json",
    )
    parser.add_argument(
        "--valid_data_path",
        type=Path,
        help="Path to the data(input question).",
        default="./valid.json",
    )
    # data
    parser.add_argument("--max_len", type=int, default=512)
    # optimizer
    parser.add_argument("--lr", type=float, default=5e-5)
    # data loader
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=2)
    # bert model
    parser.add_argument("--model_type", type=str, default="hfl/chinese-macbert-large")
    parser.add_argument("--model_dir", type=Path, default="/tmp2/jonathanhsu/ADL/model_dir")
    parser.add_argument("--log_dir", type=Path, default="/tmp2/jonathanhsu/ADL/log_dir")
    
    args = parser.parse_args()
    return args 


if __name__ == "__main__":
    args = parse_args()
    main(args)