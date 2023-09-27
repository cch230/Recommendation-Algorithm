import random
import sys

import annoy
import numpy as np
import pandas as pd
import pymysql

sys.path.append("../dbInfo/")
from DBconnect import *

# CSV 파일 읽어오기
df = pd.read_csv('USER_CTGY_RECO.csv', dtype=np.float32)

# 벡터 데이터 생성
vec = df['category'].values
data = df.drop(['category'], axis=1).values
findSetData = df.values


# Annoy 인덱스의 정확도를 최적화하는 함수 정의
def findKneighbors(num):
    annoy_index = annoy.AnnoyIndex(data.shape[1], 'angular')
    annoy_index.load(f'USER_CTGY_RECO.ann')
    print(vec.astype(int))
    print("____________________________________")
    # 'category' 열 값이 num인 행의 인덱스 찾기
    find_index = np.where(vec.astype(int) == num)[0]
    # 첫 번째 인덱스만 선택
    first_index = find_index[0] if len(find_index) > 0 else None
    print(first_index)
    # 선택한 인덱스에 해당하는 행 추출
    random_row = data[first_index]
    # print(random_index+2)
    # print(random_row)

    indices, distances = annoy_index.get_nns_by_vector(random_row, 10, include_distances=True)
    # print(indices)
    print("____________________________________")

    with pymysql.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=DATABASE,
            charset=CHARSET
    ) as conn:
        with conn.cursor() as cur:
            query = """
                              SELECT name
                              FROM tb_category
                              WHERE
                                 categoryNo  = %s
                              """

            values = (round(vec[first_index + 2]))
            cur.execute(query, values)
            for row in cur.fetchall():
                print(num, row[0])
    print("____________________________________")

    real_indices = [x + 2 for x in indices]
    for real_indice in real_indices:

        with pymysql.connect(
                host=HOST,
                port=PORT,
                user=USER,
                password=PASSWORD,
                database=DATABASE,
                charset=CHARSET
        ) as conn:
            with conn.cursor() as cur:
                query = """
                                  SELECT name
                                  FROM tb_category
                                  WHERE
                                     categoryNo  = %s
                                  """

                values = (round(vec[real_indice]))
                cur.execute(query, values)
                for row in cur.fetchall():
                    print(row[0])


# BayesianOptimization 객체 생성 및 범위 설정
# for i in range(25,31):
findKneighbors(457)
