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
"""DR.SPIDER: A DIAGNOSTIC EVALUATION BENCH- MARK TOWARDS TEXT-TO-SQL ROBUSTNESS"""


import json
import os
from typing import List, Generator, Any, Dict, Tuple
from .get_tables import dump_db_json_schema
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{chang2023dr,
  title={Dr. Spider: A Diagnostic Evaluation Benchmark towards Text-to-SQL Robustness},
  author={Chang, Shuaichen and Wang, Jun and Dong, Mingwen and Pan, Lin and Zhu, Henghui and Li, Alexander Hanbo and Lan, Wuwei and Zhang, Sheng and Jiang, Jiarong and Lilien, Joseph and others},
  journal={arXiv preprint arXiv:2301.08881},
  year={2023}
}
"""

_DESCRIPTION = """\
Dr.Spider is a comprehensive robustness benchmark based on Spider, a cross-domain text-to-SQL benchmark, to diagnose the model robustness.
"""

_HOMEPAGE = "https://github.com/awslabs/diagnostic-robustness-text-to-sql"

_LICENSE = "CC BY-SA 4.0"

_URL = "drspider.zip"

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

class drspider(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="drspider",
            version=VERSION,
            description="DR.SPIDER: A DIAGNOSTIC EVALUATION BENCH- MARK TOWARDS TEXT-TO-SQL ROBUSTNESS",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.include_train_others: bool = kwargs.pop("include_train_others", False)

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
                name="DB_schema_abbreviation",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/DB_schema_abbreviation/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/DB_schema_abbreviation/database_post_perturbation"),
                },
            ),
            datasets.SplitGenerator(
                name="DB_DBcontent_equivalence",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/DB_DBcontent_equivalence/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/DB_DBcontent_equivalence/database_post_perturbation"),
                },
            ),
            datasets.SplitGenerator(
                name="DB_schema_synonym",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/DB_schema_synonym/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/DB_schema_synonym/database_post_perturbation"),
                },
            ),
            datasets.SplitGenerator(
                name="NLQ_column_attribute",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/NLQ_column_attribute/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/NLQ_column_attribute/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="NLQ_column_carrier",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/NLQ_column_carrier/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/NLQ_column_carrier/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="NLQ_column_value",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/NLQ_column_value/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/NLQ_column_value/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="NLQ_column_synonym",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/NLQ_column_synonym/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/NLQ_column_synonym/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="NLQ_keyword_carrier",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/NLQ_keyword_carrier/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/NLQ_keyword_carrier/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="NLQ_keyword_synonym",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/NLQ_keyword_synonym/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/NLQ_keyword_synonym/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="NLQ_multitype",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/NLQ_multitype/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/NLQ_multitype/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="NLQ_others",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/NLQ_others/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/NLQ_others/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="NLQ_value_synonym",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/NLQ_value_synonym/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/NLQ_value_synonym/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="SQL_comparison",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/SQL_comparison/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/SQL_comparison/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="SQL_DB_number",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/SQL_DB_number/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/SQL_DB_number/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="SQL_DB_text",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/SQL_DB_text/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/SQL_DB_text/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="SQL_NonDB_number",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/SQL_NonDB_number/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/SQL_NonDB_number/databases"),
                },
            ),
            datasets.SplitGenerator(
                name="SQL_sort_order",
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "drspider/SQL_sort_order/questions_post_perturbation.json")],
                    "db_path": os.path.join(downloaded_filepath, "drspider/SQL_sort_order/databases"),
                }
            )
        ]

    def _generate_examples(
        self, data_filepaths: List[str], db_path: str
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function returns the examples in the raw (text) form."""
        for data_filepath in data_filepaths:
            logger.info("generating examples from = %s", data_filepath)
            with open(data_filepath, encoding="utf-8") as f:
                spider = json.load(f)
                for idx, sample in enumerate(spider):
                    db_id = sample["db_id"]
                    schema = dump_db_json_schema(
                        db=os.path.join(db_path, db_id, f"{db_id}.sqlite"), f=db_id
                    )
                    table_list = []
                    sample["query"] = sample["query"][:-1] if sample["query"][-1] == ';' else sample["query"]
                    query_tok_list = [x.lower() for x in sample["query"].split(' ')]
                    for tab_name in schema["table_names_original"]:
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
