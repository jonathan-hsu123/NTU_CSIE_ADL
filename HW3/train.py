from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import json 
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List
from datasets import Dataset
import numpy as np
from transformers.data.data_collator import DataCollatorForSeq2Seq

def open_json(path):
    with open(path, 'r') as f:
        json_list = list(f)
    return_dict = {"text": [], "id": []}
    for data in json_list:
        d = json.loads(data)
        for key, value in d.items():
            if key == 'maintext':
                return_dict['text'].append(value)
            elif key == 'id':
                return_dict['id'].append(value)
            elif key == 'title':
                if 'title' not in return_dict:
                    return_dict.update({'title': []})
                return_dict['title'].append(value)
    return return_dict

def main(args):
    train_dict = open_json(args.train_data_path)
    valid_dict = open_json(args.valid_data_path)
    train_dataset = Dataset.from_dict(train_dict)
    valid_dataset = Dataset.from_dict(valid_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    def map_data(input_data: Dict[str, List]) -> Dict[str, List]:
        return_data = {}
        for i, text in enumerate(input_data['text']):
            text = input_data['text'][i]
            tokens = tokenizer("summarize: " + text, padding='max_length', max_length=args.max_len, truncation=True)
            if i == 0:
                return_data.update({key: [] for key in tokens.keys()})
                return_data.update({'labels': []})
            for key, item in tokens.items():
                return_data[key].append(item)
            if 'title' in input_data:
                title = input_data['title'][i]
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(title, padding='max_length', max_length=args.max_len, truncation=True)
                labels['input_ids'] = [
                    (l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]
                ]
                return_data['labels'].append(labels['input_ids'])
        # print(return_data)
        return return_data
    train_dataset = train_dataset.map(map_data, batched=True, num_proc=5, remove_columns=['text', 'id', 'title'])
    # print(train_dataset)
    valid_dataset = valid_dataset.map(map_data, batched=True, num_proc=5, remove_columns=['text', 'id', 'title'])

    seed = 7789

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_dir, # output directory where model predictions and checkpoint will be saved 
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=128,
        logging_dir=args.log_dir,
        logging_strategy='steps',
        logging_steps=50,
        seed=seed,
        save_strategy='steps',
        save_steps=200,
        evaluation_strategy='steps',
        eval_steps=200,
        learning_rate=args.lr,
        fp16=True,
        num_train_epochs=args.num_epoch,
        # gradient_checkpointing=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_type)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.evaluate()
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        type=Path,
        help="Path to the data.",
        default="./data/train.jsonl",
    )
    parser.add_argument(
        "--valid_data_path",
        type=Path,
        help="Path to the data.",
        default="./data/public.jsonl",
    )
    # data
    parser.add_argument("--max_len", type=int, default=128)
    # optimizer
    parser.add_argument("--lr", type=float, default=5e-5)
    # data loader
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=20)
    # bert model
    parser.add_argument("--model_type", type=str, default="google/mt5-small")
    parser.add_argument("--model_dir", type=Path, default="/tmp2/jonathanhsu/ADL/Hw3/T5/model_dir")
    parser.add_argument("--log_dir", type=Path, default="/tmp2/jonathanhsu/ADL/Hw3/T5/log_dir")
    
    args = parser.parse_args()
    return args 



if __name__ == "__main__":
    args = parse_args()
    main(args)
