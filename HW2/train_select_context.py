from transformers import (
    AutoModelForMultipleChoice,
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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    correct_numbers = 0
    for pred, label in zip(preds, labels):
        if pred == label:
            correct_numbers += 1

    return {'accuracy': correct_numbers / len(labels)}

def open_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)

def json_to_dict(json_file, context):
    return_dict = {key: [] for key in json_file[0].keys()}
    if 'relevant' in return_dict:
        return_dict.pop('relevant', None)
        return_dict.update({'labels': []})
    max_para_size = 0
    for data in json_file:
        if len(data['paragraphs']) > max_para_size:
            max_para_size = len(data['paragraphs'])
    for data in json_file:
        for key, item in data.items():
            if key == 'paragraphs':
                para_size = len(item)
                return_dict[key].append([context[item[i]] if i < para_size else '' for i in range(max_para_size)])
            elif key == 'relevant':
                for i, pid in enumerate(data['paragraphs']):
                    if pid == data['relevant']:
                        pos = i
                return_dict['labels'].append(pos)
            else:
                 return_dict[key].append(item)

    return return_dict

def main(args):
    train_json = open_json(args.train_data_path)
    valid_json = open_json(args.valid_data_path)
    context = open_json(args.context_path)

    train_dict = json_to_dict(train_json, context)
    valid_dict = json_to_dict(valid_json, context)

    train_dataset = Dataset.from_dict(train_dict)
    valid_dataset = Dataset.from_dict(valid_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    def map_data(input_data: Dict[str, List]) -> Dict[str, List]:
        return_data = {}
        for i, paragraphs in enumerate(input_data['paragraphs']):
            question = input_data['question'][i]
            tokens = tokenizer([[question, paragraph] for paragraph in paragraphs], padding='max_length', max_length=args.max_len, truncation='only_second', return_tensors='pt')
            if i == 0:
                return_data.update({key: [] for key in tokens.keys()})
            for key, item in tokens.items():
                return_data[key].append(item)
        return return_data

    train_dataset = train_dataset.map(map_data, batched=True, num_proc=10, remove_columns=['question', 'paragraphs', 'answer', 'id'])
    valid_dataset = valid_dataset.map(map_data, batched=True, num_proc=10, remove_columns=['question', 'paragraphs', 'answer', 'id'])
    # print(train_dataset)

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
    model = AutoModelForMultipleChoice.from_pretrained(args.model_type)

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