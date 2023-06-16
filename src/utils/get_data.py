import os
import json
from get_tables import dump_db_json_schema
from bridge_content_encoder import get_database_content
from tqdm import tqdm
from process_sql import tokenize, get_schema, get_tables_with_alias, Schema, get_sql
from collections import Counter
import re
import random
import argparse

def get_structure(sql):
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


table_str_without_fk = "{table}({columns})"
table_str_with_fk = "{table}({columns}) {fks}"
fk_sep = ", "
fks = "FOREIGN KEY: {table1}.{column1} = {table2}.{column2}"
column_sep = ", "
column_str_with_values = "{column} [{values}]"
column_str_without_values = "{column}"
value_sep = ", "
question_str = 'Question: {question}\n'

def get_column_str(question: str, db_path: str ,db_id: str, table_name: str, column_name: str) -> str:
    column_name_str = column_name.lower()
    matches, type_ = get_database_content(
        question=question,
        table_name=table_name,
        column_name=column_name,
        db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
    )    

    if matches:
        if str(type_) == "<class 'str'>":
            matches = [f'\"{x}\"' for x in matches]
        else:
            matches = [str(x) for x in matches]
        string = column_str_with_values.format(column=column_name_str, values=value_sep.join(matches))
        return string
    else:
        return column_str_without_values.format(column=column_name_str)

def get_data_spider(db_path, data_filepath):
    test_data = []
    with open(data_filepath, encoding="utf-8") as f:
        data = json.load(f)
        schema_cache = dict()
        for idx, sample in tqdm(enumerate(data)):
            db_id = sample["db_id"]
            if db_id not in schema_cache:
                schema_cache[db_id] = dump_db_json_schema(
                    db=os.path.join(db_path, db_id, f"{db_id}.sqlite"), f=db_id
                )
            schema = schema_cache[db_id]
            fk_dict = {}
            for relation in schema['foreign_keys']:
                table1, column1 = relation[0]
                table2, column2 = relation[1]
                if table1 in fk_dict:
                    fk_dict[table1].append([column1.lower(), table2.lower(), column2.lower()])
                else:
                    fk_dict[table1] = [[column1.lower(), table2.lower(), column2.lower()]]
            db_table_names = schema["table_names_original"]
            db_column_names = {"table_id": [], "column_name": []}
            for item in schema['column_names_original'][1:]:
                db_column_names["table_id"].append(item[0])
                db_column_names["column_name"].append(item[1])
            question = sample["question"]
            query = sample["query"]
            query = query[:-1] if query[-1] == ';' else query
            tables = {}
            for table_id, table_name in enumerate(db_table_names):
                if table_name.lower() not in fk_dict:
                    table_str = table_str_without_fk.format(
                        table=table_name.lower(),
                        columns=column_sep.join(
                            map(
                                lambda y: get_column_str(question=question, db_path=db_path,db_id=db_id, table_name=table_name, column_name=y[1]),
                                filter(
                                    lambda y: y[0] == table_id,
                                    zip(
                                        db_column_names["table_id"],
                                        db_column_names["column_name"],
                                    ),
                                ),
                            )
                        ),
                    )
                else:
                    table_str = table_str_with_fk.format(
                        table=table_name.lower(),
                        columns=column_sep.join(
                            map(
                                lambda y: get_column_str(question=question, db_path=db_path, db_id=db_id, table_name=table_name, column_name=y[1]),
                                filter(
                                    lambda y: y[0] == table_id,
                                    zip(
                                        db_column_names["table_id"],
                                        db_column_names["column_name"],
                                    ),
                                ),
                            )
                        ),
                        fks=fk_sep.join([
                            fks.format(
                                table1=table_name,
                                column1=item[0],
                                table2=item[1],
                                column2=item[2],
                            ) for item in fk_dict[table_name]
                        ])
                    )
                tables[table_name] = table_str
            new_item = {}
            new_item['db_id'] = db_id
            new_item['tables'] = tables
            new_item['question'] = question_str.format(question=question)
            new_item['label'] = query
            test_data.append(new_item)
    return test_data

def get_data_kaggledbqa(db_path, data_filepath, table_path):
    test_data = []
    with open(table_path) as f:
        db_infos = {}
        for item in json.load(f):
            db_infos[item['db_id']] = item
        
    with open(data_filepath, encoding="utf-8") as f:
        data = json.load(f)
        schema_cache = dict()
        for idx, sample in tqdm(enumerate(data)):
            db_id = sample["db_id"]
            schema = db_infos[db_id]
            table_names = schema['table_names_original']
            column_names_origin = schema['column_names_original']
            column_types = schema['column_types']
            ## get foreign keys
            fk_dict = {}
            for relation in schema['foreign_keys']:
                idx1, idx2 = relation
                column1 = column_names_origin[idx1][1]
                table1 = table_names[column_names_origin[idx1][0]]
                column2 = column_names_origin[idx2][1]
                table2 = table_names[column_names_origin[idx2][0]]
                if table1 in fk_dict:
                    fk_dict[table1].append([column1.lower(), table2.lower(), column2.lower()])
                else:
                    fk_dict[table1] = [[column1.lower(), table2.lower(), column2.lower()]]
            ## get db_column_names 
            db_column_names = {"table_id": [], "column_name": [], "column_type": []}
            for item, col_type in zip(column_names_origin[1:], column_types[1:]):
                db_column_names["table_id"].append(item[0])
                db_column_names["column_name"].append(item[1])
                db_column_names["column_type"].append(col_type)
            question = sample["question"]
            query = sample["query"]
            query = query[:-1] if query[-1] == ';' else query
            tables = {}
            for table_id, table_name in enumerate(table_names):
                column_strs = []
                for y in zip(db_column_names["table_id"],db_column_names["column_name"],db_column_names["column_type"]):
                    if y[0] == table_id:
                        column_str = get_column_str(question=question, db_path=db_path, db_id=db_id, table_name=table_name, column_name=y[1])
                        column_strs.append(column_str)

                if table_name.lower() not in fk_dict:
                    table_str = table_str_without_fk.format(
                        table=table_name.lower(),
                        columns=column_sep.join(
                            column_strs
                        ),
                    )
                else:
                    table_str = table_str_with_fk.format(
                        table=table_name.lower(),
                        columns=column_sep.join(
                            column_strs
                        ),
                        fks=fk_sep.join([
                            fks.format(
                                table1=table_name,
                                column1=item[0],
                                table2=item[1],
                                column2=item[2],
                            ) for item in fk_dict[table_name]
                        ])
                    )
                
                tables[table_name] = table_str
            new_item = {}
            new_item['db_id'] = db_id
            new_item['tables'] = tables
            new_item['question'] = question_str.format(question=question)
            new_item['label'] = query
            test_data.append(new_item)
    return test_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set_name', dest='test_set_name', type=str, help="")

    args = parser.parse_args()
    if args.test_set_name  == "kaggledbqa":
        test_data = get_data_kaggledbqa(
            db_path=os.path.join(os.path.abspath('.'), "data/kaggledbqa/databases"), 
            data_filepath=os.path.join(os.path.abspath('.'), "data/kaggledbqa/test.json"), 
            table_path=os.path.join(os.path.abspath('.'), "data/kaggledbqa/KaggleDBQA_tables.json"),
            )
        save_file = os.path.join(os.path.abspath('.'), f'data/{args.test_set_name}/test_with_template.json')
    elif args.test_set_name[:2] == "DB":
        test_data = get_data_spider(
            db_path=os.path.join(os.path.abspath('.'), f"data/drspider/{args.test_set_name}/database_post_perturbation"),
            data_filepath=os.path.join(os.path.abspath('.'), f"data/drspider/{args.test_set_name}/questions_post_perturbation.json")            
        )
        save_file = os.path.join(os.path.abspath('.'), f'data/drspider/{args.test_set_name}/test_with_template.json')
    elif args.test_set_name[:3] in ["NLQ", "SQL"]:
        test_data = get_data_spider(
            db_path=os.path.join(os.path.abspath('.'), f"data/drspider/{args.test_set_name}/databases"),
            data_filepath=os.path.join(os.path.abspath('.'), f"data/drspider/{args.test_set_name}/questions_post_perturbation.json")         
        )
        save_file = os.path.join(os.path.abspath('.'), f'data/drspider/{args.test_set_name}/test_with_template.json')

    with open(os.path.join(os.path.abspath('.'), f'experimental_outputs/inference/{args.test_set_name}/align_select-structure.json')) as f:
        for idx, item in enumerate(json.load(f)):
            test_data[idx]['pred_structure'] = item['align_structure']
            # test_data[idx]['pred_structure'] = item['structure_candidates'][0]
            test_data[idx]['gold_structure'] = item['gold_structure']
            test_data[idx]['pred_select'] = item['align_select']
            # test_data[idx]['pred_select'] = item['select_candidates'][0]
            test_data[idx]['gold_select'] = item['gold_select']
            test_data[idx]['gold_table_list'] = item['gold_table']
            test_data[idx]['pred_table_list'] = item['table_candidates'][:2]

    with open(save_file, 'w') as f:
        json.dump(
            test_data,
            f,
            indent=4,
        )
    print(f"save the LLM's test file at {save_file}")
