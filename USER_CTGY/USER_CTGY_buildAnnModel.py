import random

import numpy as np
import pandas as pd
import annoy
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split

# CSV 파일 읽어오기
df = pd.read_csv('../DataSet/category_groupBy_user_20230825.csv')
df_shuffled = df.sample(frac=1, random_state=2)  # frac=1은 전체 데이터를 의미하며, random_state는 재현성을 위한 시드값입니다
data, test_data = train_test_split(df_shuffled, test_size=0.5, random_state=1)
df_shuffled.to_csv('USER_CTGY_RECO.csv', index=False)

# 벡터 데이터 생성
data = data.drop(['category'], axis=1).values
test_data = test_data.drop(['category'], axis=1).values

# Annoy 인덱스의 정확도를 최적화하는 함수 정의
def evaluate_n_trees(n_trees):
    annoy_index = annoy.AnnoyIndex(data.shape[1], 'angular')
    for i, vector in enumerate(data):
        annoy_index.add_item(i, vector)
    annoy_index.build(n_trees=n_trees, n_jobs=-1)

    annoy_index.save(f'USER_CTGY_RECO.ann')


# BayesianOptimization 객체 생성 및 범위 설정
# for i in range(25,31):
evaluate_n_trees(579)
print('끝')




