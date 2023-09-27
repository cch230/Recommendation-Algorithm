import json
import pickle
import sys
from typing import Tuple, List

import numpy as np
import pymysql
from numpy import array

sys.path.append("../../dbInfo/")
from DBconnect import *


def most_similar(mat: array, idx: int, k: int) -> Tuple[array, array]:
    vec = mat[idx]
    dists = mat.dot(vec) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(vec))
    top_idxs = np.argpartition(dists, -k)[-k:]
    dist_sort_args = dists[top_idxs].argsort()[::-1]
    return top_idxs[dist_sort_args], dists[top_idxs][dist_sort_args]


def dump_nearest(title: str, words: List[str], mat: array, k: int = 10) \
        -> List[str]:
    closeness = {}
    top_20_keys = []

    word_idx = words.index(title)

    sim_idxs, sim_dists = most_similar(mat, word_idx, k + 1)
    words_a = np.array(words)
    sort_args = np.argsort(sim_dists)[::-1]
    words_sorted = words_a[sim_idxs[sort_args]]
    dists_sorted = sim_dists[sort_args]
    result = zip(words_sorted, dists_sorted)

    for idx, (w, d) in enumerate(result):
        if float(d) > 0.35:
            closeness[w] = d
        # # 딕셔너리의 값을 기준으로 정렬
        # closeness = {k: v for k, v in sorted(closeness.items(), key=lambda item: item[1], reverse=True)}
        #
        # # 상위 20개 이상의 값만 남기고 나머지 삭제
    closeness = {k: closeness[k] for k in list(closeness)[:20]}
    print('연관 키워드와 정확도: ',closeness)
    top_20_items = closeness.items()

    # 상위 30개 아이템의 키를 리스트로 추출
    top_20_keys = [item[0] for item in top_20_items]

    with open(f'keyword/{title}.dat', 'wb') as f:
        pickle.dump(closeness, f)

    return top_20_keys


def get_nearest(title: str, words: List[str], mat: array) -> List[str]:
    # print(f"getting nearest words for {title}")
    try:
        with open(f'keyword/{title}.dat', 'rb') as f:
            closeness = pickle.load(f)
            print('연관 키워드와 정확도: ', closeness)
            return closeness
    except FileNotFoundError:
        return dump_nearest(title, words, mat)


print("loading valid nearest")
with open('../DataSet/valid_nearest_ko.dat', 'rb') as f:
    valid_nearest_words, valid_nearest_vecs = pickle.load(f)

loaded_data = []

with open('../DataSet/output_oneElement.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        loaded_data.append(line.strip())
print(len(loaded_data))
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
        find_word = '콜라'
        print('검색 키워드: ',find_word)
        nearest = get_nearest(find_word, valid_nearest_words, valid_nearest_vecs)
        joined_string = '|'.join(nearest)  # 리스트의 각 요소를 공백으로 구분하여 문자열로 합침

        query = """
                               SELECT DISTINCT name
                               FROM tb_category
                               WHERE name REGEXP %s 
                               limit 20;
                           """

        # Pass joined_string and nearest as parameters
        cur.execute(query, (joined_string))

        token_list = []
        for row in cur.fetchall():
            if find_word != row[0]:
                token_list.append(row[0])

        print('연관 카테고리: ', token_list)
        # JSON 파일로 저장
        with open(f'category/{find_word}.json', 'w', encoding='utf-8') as json_file:
            json.dump(token_list, json_file, ensure_ascii=False, indent=4)
