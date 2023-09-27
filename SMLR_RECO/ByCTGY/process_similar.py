import json
import operator
import os
import pickle
from typing import Tuple, List

import numpy as np
import pymysql
from numpy import array
import sys
sys.path.append("../../dbInfo/")
from DBconnect import *

def most_similar(mat: array, idx: int, k: int) -> Tuple[array, array]:
    vec = mat[idx]
    dists = mat.dot(vec) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(vec))
    top_idxs = np.argpartition(dists, -k)[-k:]
    dist_sort_args = dists[top_idxs].argsort()[::-1]
    return top_idxs[dist_sort_args], dists[top_idxs][dist_sort_args]


def dump_nearest(title: str, values: List[str], words: List[str], mat: array, k: int = 100) \
        -> List[str]:
    closeness = {}
    top_20_keys = []

    for value in values:
        try:
            word_idx = words.index(value)
        except:
            continue
        sim_idxs, sim_dists = most_similar(mat, word_idx, k + 1)
        words_a = np.array(words)
        sort_args = np.argsort(sim_dists)[::-1]
        words_sorted = words_a[sim_idxs[sort_args]]
        dists_sorted = sim_dists[sort_args]
        result = zip(words_sorted, dists_sorted)

        for idx, (w, d) in enumerate(result):
            if float(d) > 0.5:
                closeness[w] = d

            # with open(f'../DataSet/near/{title}.dat', 'wb') as f:
            #     pickle.dump(closeness, f)

        top_20_items = list(dict(sorted(closeness.items(), key=operator.itemgetter(1), reverse=True)).items())[:20]

        # 상위 30개 아이템의 키를 리스트로 추출
        top_20_keys = [item[0] for item in top_20_items]

    # print(top_20_keys)
    return top_20_keys


def get_nearest(title: str, values: List[str], words: List[str], mat: array) -> List[str]:
    # print(f"getting nearest words for {title}")
    try:
        with open(f'../DataSet/near/{title}.dat', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return dump_nearest(title, values, words, mat)


print("loading valid nearest")
with open('../DataSet/CTGY_nearest_ko.dat', 'rb') as f:
    valid_nearest_words, valid_nearest_vecs = pickle.load(f)
with open('../DataSet/output.json', 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)
print("initializing nearest words for solutions")



with pymysql.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=DATABASE,
        charset=CHARSET
) as conn:
    with conn.cursor() as cur:
        result_dict = {}
        idx = 0
        idx_list = []
        for key, values in loaded_data.items():
            idx += 1
            ### nearest 빈거 체크 !!!
            nearest = get_nearest(key.replace('/', '_'), values, valid_nearest_words, valid_nearest_vecs)

            if len(nearest) == 0:
                print(key)
                continue

            joined_string = '|'.join(nearest)  # 리스트의 각 요소를 공백으로 구분하여 문자열로 합침

            query = """
                           SELECT DISTINCT name
                           FROM tb_category
                           WHERE name REGEXP %s AND name NOT IN ({})
                           Limit 20;
                       """.format(', '.join(['%s'] * len(nearest)))

            # Pass joined_string and nearest as parameters
            cur.execute(query, (joined_string, *nearest))

            token_list = []
            for row in cur.fetchall():
                if key != row[0]:
                    token_list.append(row[0])

            if len(token_list) == 0:
                print(key)
                continue

            result_dict[key] = token_list

# JSON 파일로 저장
with open('../DataSet/relCategory.json', 'w', encoding='utf-8') as json_file:
    json.dump(result_dict, json_file, ensure_ascii=False, indent=4)


