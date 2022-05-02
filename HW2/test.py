import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
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
import numpy as np
from tqdm import tqdm

def open_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)

def json_to_dict_context(json_file, context):
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
            else:
                 return_dict[key].append(item)

    return return_dict

def main(args):
    seed = 7784
    test_json = open_json(args.test_data_path)
    context = open_json(args.context_path)
    test_dict = json_to_dict_context(test_json, context)
    test_dataset = Dataset.from_dict(test_dict)
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
    test_dataset = test_dataset.map(map_data, batched=True, num_proc=10, remove_columns=['question', 'paragraphs', 'id'])

    context_model = AutoModelForMultipleChoice.from_pretrained(args.context_model)
    training_args = TrainingArguments(
        output_dir=args.output_dir, 
        do_train=False,
        seed=seed,
    )
    trainer = Trainer(
        model=context_model,
        args=training_args,
    )
    context_result = trainer.predict(test_dataset)
    predict_labels = np.argmax(context_result[0], axis=-1)

    # QA

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    def json_to_dict_QA(json_file):
        return_dict = {}
        tokens_range = []
        total_len = 0
        for i, instance in enumerate(json_file):
            paragraph = context[instance['paragraphs'][predict_labels[i]]]
            question = instance['question']
            tokens = tokenizer(question, paragraph, padding='max_length', max_length=512, truncation='only_second', return_offsets_mapping=True, return_overflowing_tokens=True, stride=128)
            if i == 0:
                return_dict.update({key: [] for key in tokens.keys()})
            tmp = []
            for id in range(len(tokens['input_ids'])):
                tmp.append({k : v[id] for k, v in tokens.items()})
            tokens = tmp
            tokens_range.append((total_len, total_len + len(tokens)))
            total_len += len(tokens)
            for token_index, token in enumerate(tokens):
                for k, v in token.items():
                    return_dict[k].append(v)
        return return_dict, tokens_range
    QA_test_dict, test_token_range = json_to_dict_QA(test_json)
    test_dataset = Dataset.from_dict(QA_test_dict)
    QA_model = AutoModelForQuestionAnswering.from_pretrained(args.QA_model)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=False,
        seed=seed,
    )

    trainer = Trainer(
        model=QA_model,
        args=training_args,
    )
    QA_result = trainer.predict(test_dataset)
    test_dataset = Dataset.from_dict(QA_test_dict)
    QA_logits = QA_result[0]
    start_logits = QA_logits[0]
    end_logits = QA_logits[1]
    start_pos = np.argmax(start_logits, axis=-1)
    end_pos = np.argmax(end_logits, axis=-1)
    offset_mapping = [instance['offset_mapping'] for instance in test_dataset]

    answer = {}
    for qid, (token_l, token_r) in enumerate(tqdm(test_token_range)):
        start_candidate, end_candidate = [], []
        for i in range(token_l, token_r):
            if start_pos[i] != 0:
                start_candidate.append((start_logits[i][start_pos[i]], offset_mapping[i][start_pos[i]][0]))
            if end_pos[i] != 0:
                end_candidate.append((end_logits[i][end_pos[i]], offset_mapping[i][end_pos[i]][1]))

        st, ed = 0, 0
        if len(start_candidate) > 0:
            max_v = start_candidate[0][0]
            st = start_candidate[0][1]
            for i in range(len(start_candidate)):
                if start_candidate[i][0] > max_v:
                    max_v = start_candidate[i][0]
                    st = start_candidate[i][1]
        
        if len(end_candidate) > 0:
            max_v = end_candidate[0][0]
            ed = end_candidate[0][1]
            for i in range(len(end_candidate)):
                if end_candidate[i][0] > max_v:
                    max_v = end_candidate[i][0]
                    ed = end_candidate[i][1]
        
        question_id = test_json[qid]['id']
        if predict_labels[qid] < len(test_json[qid]['paragraphs']):
            context_id = test_json[qid]['paragraphs'][predict_labels[qid]]
            paragraph = context[context_id]
            answer.update({question_id: paragraph[st:ed]})

        else:
            answer.update({question_id: ''})

        final_answer = {}

        for question_id, text in answer.items():
            text = text.strip()
            final_answer.update({question_id: text})
        with open(args.pred_file, 'w', encoding='utf-8') as fp:
            fp.write('id,answer\n')
            for i, e in final_answer.items():
                fp.write(f"{i},{e.replace(',', '')}\n")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--context_path",
        type=Path,
        help="Path to the context file.",
        default="./context.json",
    )
    parser.add_argument(
        "--test_data_path",
        type=Path,
        help="Path to the data(input question).",
        default="./test.json",
    )
    # data
    parser.add_argument("--max_len", type=int, default=512)

    # bert model
    parser.add_argument("--model_type", type=str, default="bert-base-chinese")
    parser.add_argument("--context_model", type=Path, default="/tmp2/jonathanhsu/ADL/no_pretrain/context/model_dir/checkpoint-338")
    parser.add_argument("--QA_model", type=Path, default="/tmp2/jonathanhsu/ADL/bert_base/QA/model_dir/checkpoint-430")
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="File path to the output",
        default="/tmp2/jonathanhsu/ADL/output"
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="File path to the prediction file",
        default="./pred2.csv"
    )
    args = parser.parse_args()
    return args 


if __name__ == "__main__":
    args = parse_args()
    main(args)