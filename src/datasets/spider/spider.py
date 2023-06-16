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
"""Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks"""


import json
import os
from typing import List, Generator, Any, Dict, Tuple
from .get_tables import dump_db_json_schema
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{yu2018spider,
  title={Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  journal={arXiv preprint arXiv:1809.08887},
  year={2018}
}
"""

_DESCRIPTION = """\
Spider is a large-scale complex and cross-domain semantic parsing and text-toSQL dataset annotated by 11 college students
"""

_HOMEPAGE = "https://yale-lily.github.io/spider"

_LICENSE = "CC BY-SA 4.0"

#_URL = "https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"
_URL = "spider.zip"

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

class Spider(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="spider",
            version=VERSION,
            description="Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()
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
        print("downloaded_filepath")
        downloaded_filepath = dl_manager.download_and_extract(url_or_urls=_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepaths": [
                        os.path.join(downloaded_filepath, "spider/train_spider.json"),
                        os.path.join(downloaded_filepath, "spider/train_others.json"),
                    ]
                    if self.include_train_others
                    else [
                        os.path.join(downloaded_filepath, "spider/train_spider.json")
                    ],
                    "db_path": os.path.join(downloaded_filepath, "spider/database"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "spider/dev.json")],
                    "db_path": os.path.join(downloaded_filepath, "spider/database"),
                },
            ),
        ]

    def _generate_examples(
        self, data_filepaths: List[str], db_path: str
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function returns the examples in the raw (text) form."""
        for data_filepath in data_filepaths:
            logger.info("generating examples from = %s", data_filepath)
            with open(data_filepath, encoding="utf-8") as f:
                spider = json.load(f)
                print(f)
                for idx, sample in enumerate(spider):
                    db_id = sample["db_id"]
                    if db_id not in self.schema_cache:
                        self.schema_cache[db_id] = dump_db_json_schema(
                            db=os.path.join(db_path, db_id, f"{db_id}.sqlite"), f=db_id
                        )
                    schema = self.schema_cache[db_id]
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
