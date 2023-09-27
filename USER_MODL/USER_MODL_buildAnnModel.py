import random

import numpy as np
import pandas as pd
import annoy
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split

# CSV 파일 읽어오기
df = pd.read_csv('../DataSet/module_groupBy_user_20230825.csv')
df_shuffled = df.sample(frac=1, random_state=2)  # frac=1은 전체 데이터를 의미하며, random_state는 재현성을 위한 시드값입니다
df_shuffled.to_csv('USER_MODL_RECO.csv', index=False)

data = df_shuffled.drop(['moduleID'], axis=1).values
# 벡터 데이터 생성
# Annoy 인덱스의 정확도를 최적화하는 함수 정의
def evaluate_n_trees(n_trees):
    annoy_index = annoy.AnnoyIndex(data.shape[1], 'angular')
    for i, vector in enumerate(data):
        annoy_index.add_item(i, vector)
    annoy_index.build(n_trees=n_trees, n_jobs=-1)

    annoy_index.save('USER_MODL_RECO.ann')
    annoy_index.load('USER_MODL_RECO.ann')
    # 무작위로 인덱스 선택

    for i in range(100):

        random_index = random.randint(0, data.shape[0] - 1)  # 0부터 행의 개수 - 1 사이에서 무작위로 선택

        # 선택한 인덱스에 해당하는 행 추출
        random_row = data[random_index]
        random_row = [round(rr) if i < 3 else rr for i, rr in enumerate(random_row)]
        print(random_row)
        print(random_index)
        indices, distances = annoy_index.get_nns_by_vector(random_row, 10, include_distances=True)
        print(indices)
        print(distances)
        print('_________________________________')

# BayesianOptimization 객체 생성 및 범위 설정
evaluate_n_trees(189)





