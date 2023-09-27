import annoy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit

# CSV 파일 읽어오기
df = pd.read_csv('../DataSet/module_groupBy_user_20230825.csv')
df_shuffled = df.sample(frac=1, random_state=2)  # frac=1은 전체 데이터를 의미하며, random_state는 재현성을 위한 시드값입니다
data = df_shuffled.drop(['moduleID'], axis=1).values

print("hi")


# t-SNE를 사용하여 3차원으로 차원 축소
tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
data_3d_tsne = tsne.fit_transform(data)

# 3D 산점도 생성
fig = plt.figure(figsize=(20, 16))
ax = fig.add_subplot(111, projection='3d')

# 데이터를 3D 산점도에 플롯
ax.scatter(data_3d_tsne[:, 0], data_3d_tsne[:, 1], data_3d_tsne[:, 2], c='b', marker='o')
# X, Y, Z 축 눈금 설정 (세세하게 나누기)
ax.set_xticks(np.arange(min(data_3d_tsne[:, 0]), max(data_3d_tsne[:, 0]), step=5))
ax.set_yticks(np.arange(min(data_3d_tsne[:, 1]), max(data_3d_tsne[:, 1]), step=5))
ax.set_zticks(np.arange(min(data_3d_tsne[:, 2]), max(data_3d_tsne[:, 2]), step=5))
# X, Y, Z 축 레이블 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


plt.savefig('TSNE_3D.png', bbox_inches='tight')

# 3D 산점도 표시
plt.show()

print("hi2")


# t-SNE를 사용하여 벡터 시각화
tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
vectors_tsne = tsne.fit_transform(data)

# 시각화
plt.figure(figsize=(12, 10))
# X 축과 Y 축의 눈금 설정
plt.xticks(np.arange(-100, 101, 5))  # X 축 눈금 범위와 간격 설정
plt.yticks(np.arange(-100, 101, 5))  # Y 축 눈금 범위와 간격 설정
plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])
plt.title('t-SNE 2D Visualization of Vectors')
plt.savefig('TSNE_2D.png', bbox_inches='tight')
plt.show()