from collections import defaultdict
from functools import cache
import sqlite3

from fuzzywuzzy import fuzz
from numpy import array, nanmean
import spacy
from sentence_transformers import SentenceTransformer

_MAX_DB_VALUE_NUM = 100000


class Item:
    def __init__(self, db, tname, cname, pvalue):
        self.db = db
        self.tname = tname
        self.cname = cname
        self.pvalue = str(pvalue)
        self.value = []

    def clear_value(self):
        self.value = []

    def add_value(self, v):
        self.value.append(str(v))


class Fuzzy:

    @staticmethod
    def search(item):
        value_score_list = []
        for i in range(len(item.value)):
            score = fuzz.token_sort_ratio(item.pvalue, item.value[i])
            value_score_list.append((item.value[i], score))
        value_score_list = sorted(value_score_list, key = lambda x:x[1], reverse = True)
        return [s / 100 for _, s in value_score_list[:]], [v for v, _ in value_score_list[:]]


def tokenize(text):
    return [tok.text for tok in Word2vec.spacy_en.tokenizer(text.lower())]


@cache
def extract_words_embedding(words):
    processed_words = tokenize(words)
    if len(processed_words) == 0:
        processed_words = ["none"]

    embeddings_to_all_words = []

    for w in processed_words:
        if w in Word2vec.model:
            embeddings_to_all_words.append(Word2vec.model.get(w))

    if len(embeddings_to_all_words) == 0:
        processed_words = list(words)
        for w in processed_words:
            if w in Word2vec.model:
                embeddings_to_all_words.append(Word2vec.model.get(w))
        if len(embeddings_to_all_words) == 0:
            return None
    mean_of_word_embeddings = nanmean(embeddings_to_all_words, axis = 0)
    return mean_of_word_embeddings


class Word2vec:
    spacy_en = spacy.load("en_core_web_sm")
    model = dict()

    with open("glove.6B.200d.txt", "r") as f:
        # http://212.129.155.247/embedding/glove.6B.200d.zip
        for l in f:
            term, vector = l.strip().split(' ', 1)
            vector = array(vector.split(' '), dtype=float)
            model[term] = vector

    @staticmethod
    def cosine_similarity(v1, v2):
        from numpy import dot
        from numpy.linalg import norm
        return dot(v1, v2) / (norm(v1) * norm(v2))

    @staticmethod
    def search(item):
        oemb = extract_words_embedding(item.pvalue)
        value_score_list = []
        for i in range(len(item.value)):
            emb = extract_words_embedding(item.value[i])
            if emb is None:
                value_score_list.append((item.value[i], -1))
                continue
            score = Word2vec.cosine_similarity(oemb, emb)
            value_score_list.append((item.value[i], score))
        value_score_list = sorted(value_score_list, key = lambda x:x[1], reverse = True)
        return [s for _, s in value_score_list[:]], [v for v, _ in value_score_list[:]]


@cache
def get_sbert_embedding(value):
    return Sbert.model.encode(value)


class Sbert:
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

    @staticmethod
    def search(item):
        prediction_embedding = get_sbert_embedding(item.pvalue)
        value_embedding = get_sbert_embedding(tuple(item.value))
        value_score_list = []
        for i in range(len(item.value)):
            score = Word2vec.cosine_similarity(prediction_embedding, value_embedding[i])
            value_score_list.append((item.value[i], score))
        value_score_list = sorted(value_score_list, key = lambda x:x[1], reverse = True)
        return [s for _, s in value_score_list[:]], [v for v, _ in value_score_list[:]]


# database -> value -> table -> column
database_index = dict()
# database.table -> value -> table -> column
table_index = dict()


def search_value_for_item(item, match_method):
    if match_method == "fuzzy":
        return Fuzzy.search(item)
    elif match_method == "word2vec":
        return Word2vec.search(item)
    elif match_method == "sbert":
        return Sbert.search(item)


def search_in_column(conn, item, match_method, k):
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT DISTINCT {item.cname} FROM {item.tname}"
    )

    item.clear_value()
    for row in cursor:
        item.add_value(row[0])

    return search_value_for_item(item, match_method)


def search_in_table(conn, item, match_method, k):
    def build_invert_index_for_table(conn, database, table):

        table_name = f"{database}.{table}"
        table_index[table_name] = defaultdict(set)
        invert_index = table_index[table_name]

        cursor = conn.cursor()
        cursor.execute(
            f"PRAGMA TABLE_INFO({table});"
        )
        column_names = []
        for row in cursor.fetchall():
            column_names.append("`" + row[1] + "`")
        select_columns = ",".join(column_names)

        cursor.execute(
            f"SELECT DISTINCT {select_columns} FROM {table};"
        )
        for row in cursor.fetchall():
            for k in range(len(row)):
                invert_index[str(row[k])].add(column_names[k])

    table_name = f"{item.db}.{item.tname}"
    if table_name not in table_index.keys():
        build_invert_index_for_table(conn, item.db, item.tname)
    invert_index = table_index[table_name]

    item.clear_value()
    for value in list(invert_index.keys()):
        item.add_value(value)

    score, candidate = search_value_for_item(item, match_method)

    return_score, return_column, return_value = [], [], []
    for i in range(len(score)):
        for column in invert_index[candidate[i]]:
            if column != item.cname:
                return_score.append(score[i])
                return_column.append(column.strip('`'))
                return_value.append(candidate[i])

    return return_score, return_column, return_value


def search_in_database(conn, item, match_method, k):
    def build_invert_index_for_database(conn, database):

        if database in database_index.keys() and database_index[database] is None:
            return

        database_index[database] = defaultdict(lambda: defaultdict(set))
        invert_index = database_index[database]

        cursor = conn.cursor()

        cursor.execute(
            f"SELECT NAME FROM SQLITE_MASTER WHERE TYPE='table' ORDER BY NAME;"
        )
        table_names = []
        for row in cursor.fetchall():
            table_names.append(row[0])

        value_count = 0
        for j in range(len(table_names)):
            cursor.execute(
                f"SELECT COUNT(*) FROM {table_names[j]};"
            )
            value_count += cursor.fetchall()[0][0]
        if value_count > _MAX_DB_VALUE_NUM:
            database_index[database] = None

        for j in range(len(table_names)):
            cursor.execute(
                f"PRAGMA TABLE_INFO({table_names[j]});"
            )
            column_names = []
            for row in cursor.fetchall():
                column_names.append("`" + row[1] + "`")
            select_columns = ",".join(column_names)

            cursor.execute(
                f"SELECT DISTINCT {select_columns} FROM {table_names[j]};"
            )
            for row in cursor.fetchall():
                for k in range(len(row)):
                    invert_index[str(row[k])][table_names[j]].add(column_names[k])

    if item.db not in database_index.keys():
        build_invert_index_for_database(conn, item.db)
    invert_index = database_index[item.db]
    if invert_index is None:
        return [], [], [], []

    item.clear_value()
    for value in list(invert_index.keys()):
        if value not in item.value:
            item.add_value(value)

    score, value = search_value_for_item(item, match_method)
    return_score, return_value, return_table, return_column = [], [], [], []

    assert len(score) == len(value)

    for i in range(len(value)):
        for table in invert_index[value[i]]:
            if table != item.tname:
                for column in invert_index[value[i]][table]:
                    return_score.append(score[i])
                    return_value.append(value[i])
                    return_table.append(table)
                    return_column.append(column.strip("`"))

    return return_score, return_table, return_column, return_value


def number_check(pred_value, candidate):
    try:
        float(pred_value)
        float(candidate)
    except ValueError:
        return candidate
    return None


def value_match(db, question, gold_sql, pred_sql, table, column, pred_value, match_method, k):
    item = Item(db, table, column, pred_value)
    pred_value = str(pred_value)

    with sqlite3.connect(item.db) as conn:
        conn.text_factory = lambda x: str(x, "utf8", "ignore")
        score, candidate = search_in_column(conn, item, match_method, k)
        max_score = score[0]

        if str(candidate[0]).strip().lower() == pred_value.strip().lower():
            return table + '.' + column, None

        table_score, table_column, table_candidate = search_in_table(conn, item, "fuzzy", k)
        if len(table_score) > 0:
            if str(table_candidate[0]).strip().lower() == pred_value.strip().lower():
                return table + '.' + table_column[0], table_candidate[0]

        if max_score > 0.65:
            return table + '.' + column, number_check(pred_value, candidate[0])

        table_score, table_column, table_candidate = search_in_table(conn, item, match_method, k)
        if len(table_score) > 0:
            max_score = table_score[0]
            if max_score > 0.65:
                if number_check(pred_value, table_candidate[0]) is None:
                    return table + '.' + column, None
                return table + '.' + table_column[0], table_candidate[0]

        database_score, database_table, database_column, database_candidate = search_in_database(conn, item, match_method, k)
        if len(database_score) > 0:
            max_score = database_score[0]
            if max_score > 0.65:
                if number_check(pred_value, database_candidate[0]) is None:
                    return table + '.' + column, None
                return database_table[0] + '.' + database_column[0], database_candidate[0]

        return table + '.' + column, number_check(pred_value, candidate[0])
