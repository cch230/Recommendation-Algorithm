import heapq
import random

import annoy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split

# CSV 파일 읽어오기
df = pd.read_csv('../DataSet/module_groupBy_user_20230825.csv', dtype=np.float32)
df_shuffled = df.sample(frac=1, random_state=24)  # frac=1은 전체 데이터를 의미하며, random_state는 재현성을 위한 시드값입니다
data, test_data = train_test_split(df_shuffled, test_size=0.3, random_state=42)
# data.to_csv('USER_MODL_RECO.csv', index=False)

data = data.drop(['moduleID'], axis=1).values
# 벡터 데이터 생성
test_data = test_data.drop(['moduleID'], axis=1).values


# Annoy 인덱스의 정확도를 최적화하는 함수 정의
def evaluate_n_trees(n_trees):
    n_trees = int(n_trees)
    annoy_index = annoy.AnnoyIndex(data.shape[1], 'angular')
    for i, vector in enumerate(data):
        annoy_index.add_item(i, vector)
    annoy_index.build(n_trees=n_trees, n_jobs=-1)

    # 무작위로 인덱스 선택
    random_index = random.randint(0, test_data.shape[0] - 1)  # 0부터 행의 개수 - 1 사이에서 무작위로 선택
    # 선택한 인덱스에 해당하는 행 추출
    random_row = test_data[random_index]
    _, distances = annoy_index.get_nns_by_vector(random_row, 100, include_distances=True)
    total = sum(distances)
    count = len(distances)
    average = total / count
    if average == 0:
        average = 2
    return -average


# BayesianOptimization 객체 생성 및 범위 설정
pbounds = {'n_trees': (50, 1000)}
optimizer = BayesianOptimization(f=evaluate_n_trees, pbounds=pbounds, random_state=1)

# 최적화 실행
optimizer.maximize(init_points=10, n_iter=100)

# 정확도와 n_trees 값 시각화
sorted_data = sorted(optimizer.res, key=lambda x: x['params']['n_trees'])

distance_values = [round(abs(p['target']), 5) for p in sorted_data if p['target'] > -0.02]
n_trees_values = [round(p['params']['n_trees']) for p in sorted_data if p['target'] > -0.02]
min_distance = min(distance_values)
min_distance_index = distance_values.index(min_distance)

# Set custom colors
graph_color = '#84ADD2'
point_color = '#1F2541'

plt.plot(n_trees_values, distance_values, marker='o', color=graph_color, markerfacecolor=point_color)
plt.scatter(n_trees_values[min_distance_index], min_distance, color='red', s=100, label='Lowest min_distance')

plt.xlabel('Number of Trees (n_trees)')
plt.ylabel('Distance')
plt.title('Annoy Distance vs. Number of Trees (Optimization)')
plt.legend(loc='upper right')  # 범례 위치 설정
plt.grid(True)
plt.show()

# target 값과 그에 해당하는 n_trees 값을 추출하여 튜플로 저장
target_n_trees_pairs = [(item['target'], item['params']['n_trees']) for item in optimizer.res]

# 가장 낮은 target 값을 찾음
lowest_target = heapq.nlargest(10, target_n_trees_pairs, key=lambda x: x[0])
print("가장 낮은 거리 값:", lowest_target)

normalized_data = [{'target': item['target'], 'params': {'n_trees': round(item['params']['n_trees'] / 10) * 10}} for
                   item in optimizer.res]
grouped_data = {}
for item in normalized_data:
    n_trees = item['params']['n_trees']
    if n_trees not in grouped_data:
        grouped_data[n_trees] = []
    grouped_data[n_trees].append(item['target'])

# 각 그룹의 표준 편차 계산
std_deviation_data = []
for n_trees, targets in grouped_data.items():
    std_deviation = np.std(targets)
    std_deviation_data.append({'n_trees': n_trees, 'std_deviation': std_deviation})

# 표준 편차를 기준으로 정렬
sorted_data = sorted(std_deviation_data, key=lambda x: x['n_trees'])
print("가장 낮은 표준 편차 값:", sorted_data[:5])

distance_values = [round(abs(p['std_deviation']), 5) for p in sorted_data]
n_trees_values = [round(p['n_trees']) for p in sorted_data]
min_distance = min(distance_values)
min_distance_index = distance_values.index(min_distance)

plt.plot(n_trees_values, distance_values, marker='o', color=graph_color, markerfacecolor=point_color)
plt.scatter(n_trees_values[min_distance_index], min_distance, color='red', s=100, label='Lowest min_distance')

plt.xlabel('Number of Trees (n_trees)')
plt.ylabel('Std Deviation')
plt.title('Std Deviation vs. Number of Trees (10Unit)')
plt.legend(loc='upper right')  # 범례 위치 설정
plt.grid(True)
plt.show()
