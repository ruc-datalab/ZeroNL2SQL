# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KaggleDBQA: Realistic Evaluation of Text-to-SQL Parsers"""


import json
import os
from typing import List, Generator, Any, Dict, Tuple
from .get_tables import dump_db_json_schema
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{lee-2021-kaggle-dbqa,
    title = "{KaggleDBQA}: Realistic Evaluation of Text-to-{SQL} Parsers",
    author = "Lee, Chia-Hsuan  and
      Polozov, Oleksandr  and
      Richardson, Matthew",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.176",
    pages = "2261--2273"
}
"""

_DESCRIPTION = """\
Microsoft KaggleDBQA is a cross-domain and complex evaluation dataset of real Web databases, with domain-specific data types, original formatting, and unrestricted questions. It also provides database documentation which contain rich in-domain knowledge. The nature of obscure and abbreviated column/table names makes KaggleDBQA challenging to existing Text-to-SQL parsers. For more details, please see our [paper](https://arxiv.org/abs/2106.11455).
"""

_HOMEPAGE = "https://arxiv.org/abs/2106.11455"

_LICENSE = "CC BY-SA 4.0"

_URL = "kaggledbqa.zip"

def get_structure(sql: str) -> str:
    signals = ['SELECT', 'FROM']
    if len(sql['where']) > 0:
        signals.append('WHERE')
    if len(sql['groupBy']) > 0:
        signals.append('GROUP BY')
    if len(sql['having']) > 0:
        signals.append('HAVING')
    if len(sql['orderBy']) > 0:
        signals.append('ORDER BY')
    for key in ['limit', 'intersect', 'union', 'except']:
        if sql[key] != None:
            signals.append(key.upper())
            if key in ['intersect', 'union', 'except']:
                signals = signals + get_structure(sql[key])
    return signals

class kaggledbqa(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="kaggledbqa",
            version=VERSION,
            description="KaggleDBQA: Realistic Evaluation of Text-to-SQL Parsers",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "question": datasets.Value("string"),
                "structure": datasets.features.Sequence(datasets.Value("string")),
                "table_list": datasets.features.Sequence(datasets.Value("string")),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloaded_filepath = dl_manager.download_and_extract(url_or_urls=_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepaths": [
                        os.path.join(downloaded_filepath, "kaggledbqa/train.json")
                    ],
                    "db_path": os.path.join(downloaded_filepath, "kaggledbqa/databases"),
                    "table_path": os.path.join(downloaded_filepath, "kaggledbqa/KaggleDBQA_tables.json"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "kaggledbqa/test.json")],
                    "db_path": os.path.join(downloaded_filepath, "kaggledbqa/databases"),
                    "table_path": os.path.join(downloaded_filepath, "kaggledbqa/KaggleDBQA_tables.json"),
                },
            ),
        ]

    def _generate_examples(
        self, data_filepaths: List[str], db_path: str, table_path: str
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function returns the examples in the raw (text) form."""
        for data_filepath in data_filepaths:
            print(f"generating examples from = {data_filepath}")
            with open(table_path) as f:
                db_infos = {}
                for item in json.load(f):
                    for idx in range(len(item['column_names_manually_normalized_alternative'])):
                        item['column_names_manually_normalized_alternative'][idx][1] = '_'.join(item['column_names_manually_normalized_alternative'][idx][1].split(' '))
                    db_infos[item['db_id']] = item
            with open(data_filepath, encoding="utf-8") as f:
                kaggledbqa = json.load(f)
                for idx, sample in enumerate(kaggledbqa):
                    db_id = sample["db_id"]
                    schema = db_infos[db_id]
                    table_names = schema['table_names_original']
                    table_list = []
                    sample["query"] = sample["query"][:-1] if sample["query"][-1] == ';' else sample["query"]
                    query_tok_list = [x.lower() for x in sample["query"].split(' ')]
                    for tab_name in table_names:
                        if tab_name.lower() in query_tok_list:
                            table_list.append(tab_name)
                    structure = get_structure(sample["sql"])
                    yield idx, {
                        "query": sample["query"],
                        "question": sample["question"],
                        "structure": structure,
                        "table_list": table_list,
                        "db_id": db_id,
                        "db_path": db_path,
                        "db_table_names": schema["table_names_original"],
                        "db_column_names": [
                            {"table_id": table_id, "column_name": column_name}
                            for table_id, column_name in schema["column_names_original"]
                        ],
                        "db_column_types": schema["column_types"],
                        "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                        "db_foreign_keys": [
                            {"column_id": column_id, "other_column_id": other_column_id}
                            for column_id, other_column_id in schema["foreign_keys"]
                        ],
                    }
