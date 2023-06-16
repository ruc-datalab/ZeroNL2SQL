from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from transformers.training_args import TrainingArguments
from .bridge_content_encoder import get_database_matches
import re
import os
import random
from .get_tables import dump_db_json_schema

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    val_max_time: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum allowed time in seconds for generation of one example. This setting can be used to stop "
            "generation whenever the full generation exceeds the specified amount of time."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation or test examples to this "
            "value if set."
        },
    )
    num_beams: int = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_beam_groups: int = field(
        default=1,
        metadata={
            "help": "Number of beam groups to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    diversity_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Diversity penalty to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_return_sequences: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of sequences to generate during evaluation. This argument will be passed to "
            "``model.generate``, which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "A prefix to add before every source text (useful for T5 models)."},
    )
    schema_serialization_type: str = field(
        default="peteshaw",
        metadata={"help": "Choose between ``verbose`` and ``peteshaw`` schema serialization."},
    )
    training_method: str = field(
        default="PT",
        metadata={"help": "Choose between ``PT`` and ``FT``"},
    )
    prompt_path: str = field(
        default="",
        metadata={"help": "The path to the soft prompts."},
    )
    schema_serialization_randomized: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomize the order of tables."},
    )
    schema_serialization_with_db_id: bool = field(
        default=False,
        metadata={"help": "Whether or not to add the database id to the context. Needed for Picard."},
    )
    schema_serialization_with_db_content: bool = field(
        default=True,
        metadata={"help": "Whether or not to use the database content to resolve field matches."},
    )
    normalize_query: bool = field(default=True, metadata={"help": "Whether to normalize the SQL queries."})
    target_with_db_id: bool = field(
        default=False,
        metadata={"help": "Whether or not to add the database id to the target. Needed for Picard."},
    )
    

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class DataArguments:
    dataset: str = field(
        metadata={"help": "The dataset to be used. Choose between ``spider``, ``cosql``, or ``cosql+spider``, or ``spider_realistic``, or ``spider_syn``, or ``spider_dk``."},
    )
    dataset_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "spider": "./src/datasets/spider",
            "kaggledbqa": "./src/datasets/kaggledbqa",
            "drspider": "./src/datasets/drspider",
        },
        metadata={"help": "Paths of the dataset modules."},
    )
    metric_config: str = field(
        default="both",
        metadata={"help": "Choose between ``exact_match``, ``test_suite``, or ``both``."},
    )
    #we are referencing spider_realistic to spider metrics only as both use the main spider dataset as base.
    metric_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "spider": "./src/metrics/spider",
        },
        metadata={"help": "Paths of the metric modules."},
    )
    test_suite_db_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test-suite databases."})
    data_config_file : Optional[str] = field(
        default=None,
        metadata={"help": "Path to data configuration file (specifying the database splits)"}
    )
    test_sections : Optional[List[str]] = field(
        default=None,
        metadata={"help": "Sections from the data config to use for testing"}
    )


@dataclass
class TrainSplit(object):
    dataset: Dataset
    schemas: Dict[str, dict]


@dataclass
class EvalSplit(object):
    dataset: Dataset
    examples: Dataset
    schemas: Dict[str, dict]


@dataclass
class DatasetSplits(object):
    train_split: Optional[TrainSplit]
    eval_split: Optional[EvalSplit]
    test_splits: Optional[Dict[str, EvalSplit]]
    schemas: Dict[str, dict]


def _get_schemas(examples: Dataset) -> Dict[str, dict]:
    schemas: Dict[str, dict] = dict()
    for ex in examples:
        if ex["db_id"] not in schemas:
            schemas[ex["db_id"]] = {
                "db_table_names": ex["db_table_names"],
                "db_column_names": ex["db_column_names"],
                "db_column_types": ex["db_column_types"],
                "db_primary_keys": ex["db_primary_keys"],
                "db_foreign_keys": ex["db_foreign_keys"],
            }
    return schemas


def _prepare_train_split(
    dataset: Dataset,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> TrainSplit:
    schemas = _get_schemas(examples=dataset)
    dataset = dataset.map(
        lambda ex: add_serialized_schema(
            ex=ex),
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=False,
    )
    if data_training_args.max_train_samples is not None:
        dataset = dataset.select(range(data_training_args.max_train_samples))
    column_names = dataset.column_names
    dataset = dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.max_target_length,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    return TrainSplit(dataset=dataset, schemas=schemas)

def _prepare_eval_split(
    dataset: Dataset,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> EvalSplit:

    eval_examples = dataset
    schemas = _get_schemas(examples=eval_examples)
    eval_dataset = eval_examples.map(
        lambda ex: add_serialized_schema(
            ex=ex),
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=False,
    )
    column_names = eval_dataset.column_names
    eval_dataset = eval_dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.val_max_target_length,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    return EvalSplit(dataset=eval_dataset, examples=eval_examples, schemas=schemas)


def prepare_splits(
    dataset_dict: DatasetDict,
    data_args: DataArguments,
    training_args: TrainingArguments,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> DatasetSplits:
    train_split, eval_split, test_splits = None, None, None

    if training_args.do_train:
        train_split = _prepare_train_split(
            dataset_dict["train"],
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )

    if training_args.do_eval:
        if data_args.dataset in ["spider", "kaggledbqa"]:
            split_name = "validation"
        else:
            split_name = data_args.dataset
            print(f"test {split_name} in drspider")
        eval_split = _prepare_eval_split(
            dataset_dict[split_name],
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )

    if training_args.do_predict:
        test_splits = {
            section: _prepare_eval_split(
                dataset_dict[section],
                data_training_args=data_training_args,
                add_serialized_schema=add_serialized_schema,
                pre_process_function=pre_process_function,
            )
            for section in data_args.test_sections
        }
        test_split_schemas = {}
        for split in test_splits.values():
            test_split_schemas.update(split.schemas)

    schemas = {
        **(train_split.schemas if train_split is not None else {}),
        **(eval_split.schemas if eval_split is not None else {}),
        **(test_split_schemas if test_splits is not None else {}),
    }

    return DatasetSplits(
        train_split=train_split, 
        eval_split=eval_split, 
        test_splits=test_splits, 
        schemas=schemas
    )


def normalize(query: str) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

    return comma_fix(white_space_fix(lower(query)))


def serialize_schema(
    question: str,
    db_path: str,
    db_id: str,
    db_column_names: Dict[str, str],
    db_table_names: List[str],
    db_foreign_keys: Dict[str, str],
    dataset: str,
) -> str:
    db_id_str = "Database: {db_id}. "
    table_sep = ". "
    table_str = "Table: {table}. Columns: {columns}"
    column_sep = ", "
    column_str_with_values = "c{column_id}: {column} ({values})"
    column_str_without_values = "c{column_id}: {column}"
    value_sep = ", "
    table_str_without_fk = "t{table_id}: {table}({columns})"
    table_str_with_fk = "t{table_id}: {table}({columns}) {fks}"
    fk_sep = ", "
    fks = "({column1}) refers to {table2}({column2})"
    def get_column_str(question: str, db_id: str, table_name: str, column_name: str, column_id: str) -> str:
        column_name_str = column_name.lower()
        matches, type_ = get_database_matches(
            question=question,
            table_name=table_name,
            column_name=column_name,
            db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
        )
        if matches:
            string = column_str_with_values.format(column_id=column_id, column=column_name_str, values=value_sep.join(matches))
            return string
        else:
            return column_str_without_values.format(column_id=column_id, column=column_name_str)
    if dataset in ["spider", "DB_DBcontent_equivalence", "DB_schema_abbreviation", "DB_schema_synonym", "NLQ_keyword_synonym", "NLQ_keyword_carrier", "NLQ_column_synonym", "NLQ_column_carrier","NLQ_column_attribute", "NLQ_column_value", "NLQ_value_synonym", "NLQ_multitype", "NLQ_others",
              "SQL_comparison","SQL_sort_order","SQL_NonDB_number","SQL_DB_text","SQL_DB_number"]:
        schema = dump_db_json_schema(
                    db=os.path.join(db_path, db_id, f"{db_id}.sqlite"), f=db_id
                )
        fk_dict = {}
        for relation in schema['foreign_keys']:
            table1, column1 = relation[0]
            table1_id = db_table_names.index(table1)
            column1_id = 0
            for y in zip(db_column_names["table_id"], db_column_names["column_name"]):
                if y[0] == table1_id:
                    if y[1].lower() == column1.lower():
                        break
                    column1_id += 1
            table2, column2 = relation[1]
            table2_id = db_table_names.index(table2)
            column2_id = 0
            for y in zip(db_column_names["table_id"], db_column_names["column_name"]):
                if y[0] == table2_id:
                    if y[1].lower() == column2.lower():
                        break
                    column2_id += 1
            
            if table1 in fk_dict:
                fk_dict[table1].append([f'c{column1_id}', f't{table2_id}', f'c{column2_id}'])
            else:
                fk_dict[table1] = [[f'c{column1_id}', f't{table2_id}', f'c{column2_id}']]
        tables = []
        for table_id, table_name in enumerate(db_table_names):
            column_id = 0
            column_strs = []
            for y in zip(db_column_names["table_id"], db_column_names["column_name"]):
                if y[0] == table_id:
                    column_strs.append(get_column_str(question=question, db_id=db_id, table_name=table_name, column_name=y[1], column_id=str(column_id)))
                    column_id += 1
            if table_name.lower() not in fk_dict:
                table_str = table_str_without_fk.format(
                    table_id=str(table_id),
                    table=table_name.lower(),
                    columns=column_sep.join(
                        column_strs
                    ),
                )
            else:
                table_str = table_str_with_fk.format(
                    table_id=str(table_id),
                    table=table_name.lower(),
                    columns=column_sep.join(
                        column_strs
                    ),
                    fks=fk_sep.join([
                        fks.format(
                            column1=item[0],
                            table2=item[1],
                            column2=item[2],
                        ) for item in fk_dict[table_name]
                    ])
                )
            tables.append(table_str)
    elif dataset == "kaggledbqa":
        ## get foreign keys
        fk_dict = {}
        for idx1, idx2 in zip(db_foreign_keys['column_id'], db_foreign_keys['other_column_id']):
            column1 = db_column_names['column_name'][idx1]
            column2 = db_column_names['column_name'][idx2]
            table1_id = db_column_names['table_id'][idx1]
            table1 = db_table_names[table1_id]
            column1_id = 0
            for y in zip(db_column_names["table_id"], db_column_names["column_name"]):
                if y[0] == table1_id:
                    if y[1].lower() == column1.lower():
                        break
                    column1_id += 1
            table2_id = db_column_names['table_id'][idx2]
            column2_id = 0
            for y in zip(db_column_names["table_id"], db_column_names["column_name"]):
                if y[0] == table2_id:
                    if y[1].lower() == column2.lower():
                        break
                    column2_id += 1
            
            if table1 in fk_dict:
                fk_dict[table1].append([f'c{column1_id}', f't{table2_id}', f'c{column2_id}'])
            else:
                fk_dict[table1] = [[f'c{column1_id}', f't{table2_id}', f'c{column2_id}']]

        tables = []
        for table_id, table_name in enumerate(db_table_names):
            column_id = 0
            column_strs = []
            for y in zip(db_column_names["table_id"], db_column_names["column_name"]):
                if y[0] == table_id:
                    column_strs.append(get_column_str(question=question, db_id=db_id, table_name=table_name, column_name=y[1], column_id=str(column_id)))
                    column_id += 1
            if table_name.lower() not in fk_dict:
                table_str = table_str_without_fk.format(
                    table_id=str(table_id),
                    table=table_name.lower(),
                    columns=column_sep.join(
                        column_strs
                    ),
                )
            else:
                table_str = table_str_with_fk.format(
                    table_id=str(table_id),
                    table=table_name.lower(),
                    columns=column_sep.join(
                        column_strs
                    ),
                    fks=fk_sep.join([
                        fks.format(
                            column1=item[0],
                            table2=item[1],
                            column2=item[2],
                        ) for item in fk_dict[table_name]
                    ])
                )
            tables.append(table_str)
    serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    return serialized_schema
