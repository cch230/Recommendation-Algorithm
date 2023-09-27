import random

import numpy as np
import pandas as pd
import annoy
import matplotlib.pyplot as plt
import pymysql
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split

import sys
sys.path.append("../dbInfo/")
from DBconnect import *

# CSV 파일 읽어오기
df = pd.read_csv('USER_MODL_RECO.csv', dtype=np.float32)


# 벡터 데이터 생성
vec = df['moduleID'].values
data = df.drop(['moduleID'], axis=1).values
findSetData = df.values
# Annoy 인덱스의 정확도를 최적화하는 함수 정의
def findKneighbors(num):

    annoy_index = annoy.AnnoyIndex(data.shape[1], 'angular')
    annoy_index.load('USER_MODL_RECO.ann')
    find_index = np.where(vec.astype(int) == num)[0]
    # 첫 번째 인덱스만 선택
    first_index = find_index[0] if len(find_index) > 0 else None

    # 선택한 인덱스에 해당하는 행 추출
    random_row = data[first_index]
    random_row = [round(rr) if i < 3 else rr for i, rr in enumerate(random_row)]

    # print(random_index+2)
    # print(random_row)
    indices, distances = annoy_index.get_nns_by_vector(random_row, 10, include_distances=True)
    # print(indices)

    print("____________________________________")
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
                          SELECT type 
                          FROM tb_module
                          WHERE
                             moduleNo  = %s
                          """

            values = (round(vec[first_index+2]))
            cur.execute(query, values)
            for row in cur.fetchall():
                print(row[0])
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
                              SELECT type 
                              FROM tb_module
                              WHERE
                                 moduleNo  = %s
                              """

                values = (round(vec[real_indice]))
                cur.execute(query, values)
                for row in cur.fetchall():
                    print(row[0])


# BayesianOptimization 객체 생성 및 범위 설정
findKneighbors(456)





