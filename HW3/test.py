from cgi import test
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import os
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
    return return_dict

def main(args):
    test_dict = open_json(args.test_data_path)
    test_dataset = Dataset.from_dict(test_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    def map_data(input_data: Dict[str, List]) -> Dict[str, List]:
        return_data = {}
        for i, text in enumerate(input_data['text']):
            text = input_data['text'][i]
            tokens = tokenizer("summarize: " + text, padding='max_length', max_length=args.max_len, truncation=True)
            if i == 0:
                return_data.update({key: [] for key in tokens.keys()})
            for key, item in tokens.items():
                return_data[key].append(item)
        # print(return_data)
        return_data['decoder_input_ids'] = return_data['input_ids']
        return return_data
    test_dataset = test_dataset.map(map_data, batched=True, num_proc=5)
    # print(test_dataset)
    seed = 7789
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=False,
        predict_with_generate=True,
        seed=seed,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    predict_results = trainer.predict(
        test_dataset, metric_key_prefix="predict", max_length=args.max_len,
        num_beams=args.num_beams
    )
    # print(type(predict_results))
    predictions = tokenizer.batch_decode(
        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    predictions = [pred.strip() for pred in predictions]
    output_prediction_file = os.path.join(training_args.output_dir, args.pred_file)
    with open(output_prediction_file, "w") as writer:
        for i in range(len(predictions)):
            tmp_dict = dict()
            tmp_dict.update({"title": predictions[i]})
            tmp_dict.update({"id": test_dataset['id'][i]})
            # print(tmp_dict)
            writer.write(json.dumps(tmp_dict))
            writer.write('\n')
    # 
    #     writer.write("\n".join(predictions))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_data_path",
        type=Path,
        help="Path to the data.",
        default="./data/public.jsonl",
    )
    # data
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=16)
    # bert model
    parser.add_argument("--model_type", type=str, default="google/mt5-small")
    parser.add_argument("--output_dir", type=Path, default="./")
    parser.add_argument("--pred_file", type=Path, default="pred.jsonl")
    parser.add_argument("--model_path", type=Path, default="/tmp2/jonathanhsu/ADL/Hw3/T5/model_dir/checkpoint-3200")
    
    args = parser.parse_args()
    return args 



if __name__ == "__main__":
    args = parse_args()
    main(args)