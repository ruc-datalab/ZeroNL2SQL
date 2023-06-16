import os
import json
import numpy as np
from typing import Optional
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from .dataset import (
    DataArguments,
    DataTrainingArguments,
    DatasetSplits,
    TrainSplit,
    _prepare_train_split,
    prepare_splits,
    normalize, 
    serialize_schema
)
from .trainer import Seq2SeqTrainer, EvalPrediction
from .process_sql import tokenize, get_schema, get_tables_with_alias, Schema, get_sql

def add_serialized_schema(ex: dict, data_training_args: DataTrainingArguments, data_args: DataArguments) -> dict:
    serialized_schema = serialize_schema(
        question=ex["question"],
        db_path=ex["db_path"],
        db_id=ex["db_id"],
        db_column_names=ex["db_column_names"],
        db_table_names=ex["db_table_names"],
        db_foreign_keys=ex["db_foreign_keys"],
        dataset=data_args.dataset,
    )
    return {"serialized_schema": serialized_schema}

def getSelect(sql: str, tables_with_alias: dict):
    select_begin = 0
    select_end = sql.lower().index(' from ')
    select = sql[select_begin:select_end].lower()
    for alias in ['t1', 't2', 't3', 't4']:
        if alias in select and alias in tables_with_alias:
            select = select.replace(alias, tables_with_alias[alias])
    return select


def pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    data_training_args: DataTrainingArguments,
    data_args: DataArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    inputs = []
    targets = []
    for db_id, db_path, question, query, serialized_schema, table_list, structure, db_table_names, db_column_names in zip(batch["db_id"], batch["db_path"], batch["question"], batch["query"], batch["serialized_schema"], batch["table_list"], batch["structure"], batch["db_table_names"], batch["db_column_names"]):
        for idx in range(len(table_list)):
            table_list[idx] = f't{db_table_names.index(table_list[idx])}'
        db = os.path.join(db_path, db_id, db_id + ".sqlite")
        tables_with_alias = get_tables_with_alias(Schema(get_schema(db)).schema, tokenize(query))
        query = query[:-1] if query[-1] == ';' else query
        select = getSelect(query, tables_with_alias)
        column_id = None
        table_id = None
        for y in zip(db_column_names["table_id"], db_column_names["column_name"]):
            if y[0] != table_id:
                table_id = y[0]
                column_id = 0
            else:
                column_id += 1
            table_name = db_table_names[y[0]]
            if f't{y[0]}' in table_list and y[1].lower() in select:
                if f' {y[1].lower()} ' in select:
                    select = select.replace(f' {y[1].lower()} ', f' t{y[0]}.c{column_id} ')
                elif f' {y[1].lower()}' == select[-len(f' {y[1].lower()}'):]:
                    select = select.replace(f' {y[1].lower()}', f' t{y[0]}.c{column_id}')
                elif f' {y[1].lower()},' in select:
                    select = select.replace(f' {y[1].lower()},', f' t{y[0]}.c{column_id},')
                elif f' {y[1].lower()})' in select:
                    select = select.replace(f' {y[1].lower()})', f' t{y[0]}.c{column_id})')
                elif f'({y[1].lower()})' in select:
                    select = select.replace(f'({y[1].lower()})', f'(t{y[0]}.c{column_id})')
                elif f'{table_name.lower()}.{y[1].lower()}' in select:
                    select = select.replace(f'{table_name.lower()}.{y[1].lower()}', f't{y[0]}.c{column_id}')
        prompts = {
            "structure": [f"Translate the question into a SQL structure according to the database. question: {question}, database: {serialized_schema}.", '-'.join(structure)],
            "select": [f"Generate the SELECT sub-clause of this question according to the database. question: {question}, database: {serialized_schema}.", select],
            "table": [f"Generate the relevant tables of this question according to the database. question: {question}, database: {serialized_schema}.", ' '.join(table_list)],
        }
        for type_ in prompts:
            inputs.append(prompts[type_][0])
            targets.append(prompts[type_][1])     
   
    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

class PromptGenTrainer(Seq2SeqTrainer):
    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        if self.args.do_train:
            real_decoded_label_ids = []
            real_predictions = []
            db_cache = {}
            for x in examples:
                if x["db_id"] not in db_cache:
                    db_cache[x["db_id"]] = {'tab_map':{}, 'tabcol_map':{}}
                    db_column_names = x["db_column_names"]
                    db_table_names = x["db_table_names"]
                    table_dict = {}
                    for y in zip(db_column_names["table_id"], db_column_names["column_name"]):
                        if y[0] >= 0:
                            if db_table_names[y[0]] in table_dict:
                                table_dict[db_table_names[y[0]]].append(y[1])
                            else:
                                table_dict[db_table_names[y[0]]] = [y[1]]
                    for tab_id in range(len(db_table_names)-1, -1, -1):
                        table_name = db_table_names[tab_id]
                        db_cache[x["db_id"]]['tab_map'][f't{tab_id}'] = table_name
                        for col_id in range(len(table_dict[table_name])-1, -1, -1):
                            column_name = table_dict[table_name][col_id]
                            db_cache[x["db_id"]]['tabcol_map'][f't{tab_id}.c{col_id}'] = f'{table_name}.{column_name}'
            
            for idx, labelpred in enumerate(zip(decoded_label_ids, predictions)):
                label, pred = labelpred
                db_id = examples[idx//3]["db_id"]
                if idx%3 == 0:
                    real_decoded_label_ids.append(label)
                    real_predictions.append(pred)
                elif idx%3 == 1:
                    tabcol_map = db_cache[db_id]["tabcol_map"]
                    for key in tabcol_map:
                        label = label.replace(key, tabcol_map[key])
                        pred = pred.replace(key, tabcol_map[key])
                    real_decoded_label_ids.append(label)
                    real_predictions.append(pred)
                elif idx%3 == 2:
                    tab_map = db_cache[db_id]["tab_map"]
                    for key in tab_map:
                        label = label.replace(key, tab_map[key])
                        pred = pred.replace(key, tab_map[key])
                    real_decoded_label_ids.append(label)
                    real_predictions.append(pred)

                    
            with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
                json.dump(
                    [dict(**{"input": input}, **{"prediction": prediction}, **{"label": label}, **{"score": prediction.lower()==label.lower()}, **{"real_pred": real_pred}, **{"real_label": real_label}) for input, prediction, label, real_pred, real_label in zip(inputs, predictions, decoded_label_ids, real_predictions, real_decoded_label_ids)],
                    f,
                    indent=4,
                )
            return EvalPrediction(predictions=predictions, label_ids=decoded_label_ids)
        else:
            real_decoded_label_ids = []
            real_predictions = []
            db_cache = {}
            for x in examples:
                if x["db_id"] not in db_cache:
                    db_cache[x["db_id"]] = {'tab_map':{}, 'tabcol_map':{}}
                    db_column_names = x["db_column_names"]
                    db_table_names = x["db_table_names"]
                    table_dict = {}
                    for y in zip(db_column_names["table_id"], db_column_names["column_name"]):
                        if y[0] >= 0:
                            if db_table_names[y[0]] in table_dict:
                                table_dict[db_table_names[y[0]]].append(y[1])
                            else:
                                table_dict[db_table_names[y[0]]] = [y[1]]
                    for tab_id in range(len(db_table_names)-1, -1, -1):
                        table_name = db_table_names[tab_id]
                        db_cache[x["db_id"]]['tab_map'][f't{tab_id}'] = table_name
                        for col_id in range(len(table_dict[table_name])-1, -1, -1):
                            column_name = table_dict[table_name][col_id]
                            db_cache[x["db_id"]]['tabcol_map'][f't{tab_id}.c{col_id}'] = f'{table_name}.{column_name}'
            hyp_dir = os.path.join(self.args.output_dir, 'hypotheses.json')
            hyp_data = json.load(open(hyp_dir))
            real_topk_preds = []
            for idx, (label, pred, hyp) in enumerate(zip(decoded_label_ids, predictions, hyp_data)):
                topk_preds = hyp["topk_preds"]
                db_id = examples[idx//3]["db_id"]
                if idx%3 == 0:
                    real_decoded_label_ids.append(label)
                    real_predictions.append(pred)
                    real_topk_preds.append(topk_preds)
                elif idx%3 == 1:
                    tabcol_map = db_cache[db_id]["tabcol_map"]
                    for key in tabcol_map:
                        label = label.replace(key, tabcol_map[key])
                        pred = pred.replace(key, tabcol_map[key])
                        for idx_ in range(len(topk_preds)):
                            topk_preds[idx_] = topk_preds[idx_].replace(key, tabcol_map[key])
                    real_decoded_label_ids.append(label)
                    real_predictions.append(pred)
                    real_topk_preds.append(topk_preds)
                elif idx%3 == 2:
                    tab_map = db_cache[db_id]["tab_map"]
                    # remove invalid table prediction
                    for idx_ in range(len(topk_preds)):
                        tabs = topk_preds[idx_].split(' ')
                        for tab in tabs:
                            if tab not in tab_map:
                                topk_preds[idx_] = ''
                                break
                    for key in tab_map:
                        label = label.replace(key, tab_map[key])
                        pred = pred.replace(key, tab_map[key])
                        for idx_ in range(len(topk_preds)):
                            topk_preds[idx_] = topk_preds[idx_].replace(key, tab_map[key])
                    real_decoded_label_ids.append(label)
                    real_predictions.append(pred)
                    real_topk_preds.append(topk_preds)

                    
            with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
                json.dump(
                    [dict(**{"input": input}, **{"prediction": prediction}, **{"label": label}, **{"score": prediction.lower()==label.lower()}, **{"real_pred": real_pred}, **{"real_label": real_label}, **{"topk_preds": topk_preds}) for input, prediction, label, real_pred, real_label, topk_preds in zip(inputs, predictions, decoded_label_ids, real_predictions, real_decoded_label_ids, real_topk_preds)],
                    f,
                    indent=4,
                )
            return EvalPrediction(predictions=predictions, label_ids=decoded_label_ids)
            
    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids = eval_prediction
        accuracy = []
        accuracy.extend(
            (
                pred.lower() == actual.lower()
                for pred, actual in zip(predictions, label_ids)
            )
        )
        eval_metric = np.mean(accuracy)
        test_suite = dict()
        return {**{"exact_match": eval_metric}, **test_suite}
