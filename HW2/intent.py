from transformers import (
    AutoModelForSequenceClassification,
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
        if p == l:
            correct_num += 1
    return {'accuracy': correct_num / len(prediction)}

def open_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)

def json_to_dict(json_file, context):
    return_dict = {key: [] for key in json_file[0].keys()}
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
    intent2idx: Dict[str, int] = json.loads(args.intent_to_idx_data_path.read_text())
    idx2intent = {idx: intent for intent, idx in intent2idx.items()}

    train_json = open_json(args.train_data_path)
    valid_json = open_json(args.valid_data_path)
    test_json = open_json(args.test_data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    def map_data(input_data) -> Dict[str, List]:
        return_data = {}
        for i, instance in enumerate(input_data):
            text = instance['text']
            tokens = tokenizer(text, padding='max_length', max_length=args.max_len)
            if i == 0:
                return_data.update({key: [] for key in tokens.keys()})
                if 'intent' in instance:
                    return_data.update({'labels': []})
            for key, values in tokens.items():
                return_data[key].append(values)
            if 'intent' in instance:
                return_data['labels'].append(intent2idx[instance['intent']])
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
    model = AutoModelForSequenceClassification.from_pretrained(args.model_type, num_labels=150)

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
    pred_intents = []

    for i, instance in enumerate(test_json):
        test_id = instance['id']
        pred_intents.append((test_id, idx2intent[predictions[i]]))
    
    with open(args.pred_file, 'w+') as f:
        f.write('id,intent\n')
        for i, e in pred_intents:
            f.write(f'{i},{e}\n')




def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        type=Path,
        help="Path to the data(input question).",
        default="/tmp2/jonathan/data/intent/train.json",
    )
    parser.add_argument(
        "--valid_data_path",
        type=Path,
        help="Path to the data(input question).",
        default="/tmp2/jonathan/data/intent/eval.json",
    )
    parser.add_argument(
        "--test_data_path",
        type=Path,
        help="Path to the data(input question).",
        default="/tmp2/jonathan/data/intent/test.json",
    )
    parser.add_argument(
        "--intent_to_idx_data_path",
        type=Path,
        help="Path to the data(input question).",
        default="/tmp2/jonathan/cache/intent/intent2idx.json",
    )
    
    # data
    parser.add_argument("--max_len", type=int, default=512)
    # optimizer
    parser.add_argument("--lr", type=float, default=5e-5)
    # data loader
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=5)
    # bert model
    parser.add_argument("--model_type", type=str, default="bert-base-cased")
    parser.add_argument("--model_dir", type=Path, default="/tmp2/jonathanhsu/ADL/bert_base/intent/model_dir")
    parser.add_argument("--log_dir", type=Path, default="/tmp2/jonathanhsu/ADL/bert_base/intent/log_dir")

    parser.add_argument(
        "--pred_file",
        type=Path,
        help="File path to the prediction file",
        default="./pred_intent.csv"
    )
    
    args = parser.parse_args()
    return args 



if __name__ == "__main__":
    args = parse_args()
    main(args)