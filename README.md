# *Recommendation-Algorithm* 

## Index 
- [Directory](#Directory)
- [DataSet](#DataSet)
- [Annoy 라이브러리를 활용한 벡터 유사도 최적화](#annoy-라이브러리를-활용한-벡터-유사도-최적화)
- [t-SNE를 활용한 벡터 데이터 시각화](#t-SNE를-활용한-벡터-데이터-시각화)
- [한국어 텍스트 분류를 위한 BERT 모델을 활용한 단어 필터링](#한국어-텍스트-분류를-위한-bert-모델을-활용한-단어-필터링)
- [konlpy와 googletrans를 활용한 카테고리 형태소 분석 및 번역](#konlpy와-googletrans를-활용한-카테고리-형태소-분석-및-번역)
- [Word2Vec 한국어 단어 임베딩 데이터베이스 구축](#word2vec-한국어-단어-임베딩-데이터베이스-구축)
- [의미론적 단어유사도를 활용한 카테고리/키워드 추천](#의미론적-단어유사도를-활용한-카테고리키워드-추천)
- [라이선스](#라이선스)


---
## Directory
- [`USER_CTGY`](USER_CTGY): 사용자 기반 협업필터링을 통한 카테고리 추천 알고리즘(User-based CF)
- [`USER_MODL`](USER_MODL): 콘텐츠 기반 협업필터링을 통한 묘듈별 기능 추천 알고리즘(Item-based CF)
- [`SMLR_RECO`](SMLR_RECO): 형태소 분석, 태깅 라이브러리를 이용한 의미론적 키워드, 카테고리 추천 알고리즘
  1. [`ByCTGY`](SMLR_RECO/ByCTGY): 카테고리 기반 연관 카테고리 추천 알고리즘
  2. [`ByKYWD`](SMLR_RECO/ByKYWD): 키워드 기반 연관 카테고리, 키워드 추천 알고리즘
  

---
## DataSet
1. 유저의 명시적 데이터와 카테고리/모듈 별 행동 기록을 분석한 암시적 피드백을 활용한 [카테고리](DataSet/category_groupBy_user_20230825.csv)/[모듈](DataSet/module_groupBy_user_20230825.csv) 벡터 데이터
   - 동적 벡터 가중치: 사용한 포인트, 더보기 요청, 검색, 30초 이상 체류, 저장/갱신 활성화, 좋아요 / 댓글
   - 정적 벡터 가중치: 모듈, 카테고리, 키워드, 연간 키워드
2. spellcheck-ko에서 제공하는 [한국어기초사전](https://krdict.korean.go.kr/), [표준국어대사전](https://stdict.korean.go.kr/), [우리말샘](https://opendict.korean.go.kr/) 기반  [한국어 맞춤법 사전](https://github.com/spellcheck-ko/hunspell-dict-ko/releases/download/0.7.92/ko-aff-dic-0.7.92.zip)
3. Facebook에서 제공하는 FastText의 300차원 벡터로 표현하여 단어의 의미적 관계를 반영한 한국어 Word2Vec 모델 [한국어 단어 벡터](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.vec.gz)
4. 네이버 카테고리를 형태소 분석하여 나눈 [네이버 카테고리 말 뭉치](SMLR_RECO/data/output_oneElement.txt)


---
## Annoy 라이브러리를 활용한 벡터 유사도 최적화
[`Annoy`](https://github.com/spotify/annoy)와 [`Bayesian Optimization`](https://github.com/bayesian-optimization/BayesianOptimization) 라이브러리를 사용하여 벡터 유사도를 최적화 합니다.  

### 사용된 주요 기술 및 라이브러리
- **Annoy 라이브러리:** 벡터 유사도를 효율적으로 계산하고 검색하기 위한 라이브러리
- **Bayesian Optimization:** 목적 함수를 최적화하기 위한 효율적인 알고리즘
- **pandas:** 데이터 조작 및 계산 라이브러리
- **numpy:** 다차원 배열을 처리하기 위한 라이브러리
- **matplotlib:** 데이터 시각화 라이브러리

### 사용법
1. **의존성 설치:**
   ```bash
   pip install annoy pandas numpy scikit-learn bayesian-optimization matplotlib

2. **코드 실행:**
   ```bash
   python *_optimizeAnnModel.py
위 명령어를 실행하여 Annoy 라이브러리를 사용한 벡터 유사도 최적화를 수행

### 파이썬 코드 파일 (*_similarity_optimization.py)에 대한 설명
- `evaluate_n_trees(n_trees)`: Annoy 인덱스의 정확도를 최적화하기 위한 함수로, 주어진 트리 수에 대해 벡터 유사도를 계산하고 최근접 이웃들의 평균 거리를 반환
- `BayesianOptimization`: 트리 수(n_trees)를 조정하여 목적 함수(evaluate_n_trees)를 최적의 평균 거리를 탐색하며 트리 수 최적화
  

---  
## t-SNE를 활용한 벡터 데이터 시각화 
scikit-learn의 [`t-SNE`](https://github.com/scikit-learn/scikit-learn/tree/main) 알고리즘을 활용하여 벡터 데이터를 시각화 합니다.

### 사용된 주요 기술 및 라이브러리
- **t-SNE:** 고차원 데이터의 구조를 유지하면서 저차원으로 축소하여 시각화하는 데 사용되는 알고리즘
- **matplotlib:** 데이터 시각화 라이브러리
- **scikit-learn:** 머신러닝 모델 구현 라이브러리
- **pandas:** 데이터 조작 및 계산 라이브러리
- **numpy:** 다차원 배열을 처리하기 위한 라이브러리

### 사용법

1. **의존성 설치:**
   ```bash
   pip install scikit-learn matplotlib pandas numpy
2. **코드 실행:**
   ```bash 
   python visualize_vectors.py
TSNE_3D.png 및 TSNE_2D.png 이미지 파일로 2D 및 3D t-SNE 결과가 생성
  

--- 
## 한국어 텍스트 분류를 위한 BERT 모델을 활용한 단어 필터링
텍스트에서 깨끗한 단어를 필터링하기 위해 자연어 처리(NLP) 작업에 사용되는 사전 훈련된 언어 모델인 BERT 모델을 사용합니다.

### 사용된 주요 기술 및 라이브러리
- **BERT:** 양방향 트랜스포머 모델을 기반으로 한 사전 훈련된 언어 모델, Smilegate-ai에서 제공하는 [`kor_unsmile`](https://github.com/smilegate-ai/korean_unsmile_dataset) 모델을 활용
- **Hugging Face Transformers:** 다국어로 된 여러 사전 훈련 모델을 제공하는 라이브러리, 모델을 로드하고 텍스트 분류를 수행

### 사용법

1. **의존성 설치:**
   ```bash
   pip install transformers tqdm

2. **사전 훈련된 BERT 모델 및 토크나이저 다운로드:**
    ```python
    from transformers import BertForSequenceClassification, AutoTokenizer
    
    model_name = 'smilegate-ai/kor_unsmile'
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
3. **단어 필터링 수행:**
    ```bash
   python filter_words.py
spellcheck-ko에서 제공하는 한국어 단어를 분류하고 깨끗한 단어를 추출하여 결과는 data/ko_filtered.txt 파일에 저장

### 파이썬 코드 파일 (filter_words.py)에 대한 설명
 - `get_predicated_label(output_labels, min_score)`: BERT 모델의 출력 레이블에서 지정된 최소 점수 이상인 레이블만을 반환하는 함수

- `TextClassificationPipeline`: 텍스트 분류 파이프라인을 초기화하고 설정. 텍스트를 입력으로 받아 BERT 모델을 사용하여 분류를 수행하고 결과를 반환
  

---
## konlpy와 googletrans를 활용한 카테고리 형태소 분석 및 번역 
[`KoNLPy`](https://github.com/konlpy/konlpy)의 여러 태깅 라이브러리를 활용하여 카테고리를 형태소 분석하여 유의미한 단어로 추출하고, [`googletrans`](https://github.com/ssut/py-googletrans)를 활용하여 추출된 단어들을 번역 한후 정규화를 거쳐 새로운 유사 단어들을 확보합니다.

### 사용된 기술 및 라이브러리
- **konlpy**: 한글 형태소 분석을 위한 라이브러리, Okt, Hannanum, Kkma, Komoran을 사용하여 형태소 분석을 수행
- **googletrans**: Google Translate API를 활용하여 단어를 번역하는 데 사용
- **re**: 정규 표현식을 사용하여 단어를 필터링하는 데 사용

### 사용법

1. **의존성 설치:**
   ```bash
   pip install konlpy googletrans

2. **카테고리 형태소 분석 및 번역을 수행:**
   ```bash
   python category_corpus.py
카테고리에서 새로운 유사단어를 추출하여, output.json 와 output_oneElement.txt 에 저장

### 파이썬 코드 파일 (category_corpus.py)에 대한 설명
- `tokenize_and_join(input_file: str) -> Tuple[List[int], List[str]]`: 입력 파일에서 각 라인을 읽어와 형태소 분석 및 번역을 수행하여 유의미한 단어를 추출하고, 이를 파일로 저장


 --- 
## Word2Vec 한국어 단어 임베딩 데이터베이스 구축
한국어 Word2Vec 임베딩 모델을 활용하여 단어 벡터를 추출하고, 저장합니다.

### 사용된 주요 기술 및 라이브러리

- **Word2Vec:** 한국어 단어의 분산 표현을 학습하기 위한 단어 임베딩 모델 기술, Facebook에서 제공하는 Word2Vec 모델을 활용하여 단어 벡터를 추출하고 사용
- **SQLite:** 경량화 DBMS 라이브러리, 단어와 그에 해당하는 벡터를 저장
- **unicodedata:** 유니코드 문자에 대한 데이터베이스를 제공하는 라이브러리
- **pickle:** `파이썬 객체를 직렬화하고 역직렬화하는 라이브러리
- **numpy:** 다차원 배열을 처리하기 위한 라이브러리

### 사용법

1. **의존성 설치:**
   ```bash
   pip install numpy tqdm

2. **한국어 Word2Vec 데이터베이스 구축:**
   ```bash
   python process_vecs_*.py
한국어 Word2Vec 모델에서 단어 벡터를 추출하여, *_guesses_ko.db 와 *_nearest_ko.dat 에 저장

### 파이썬 코드 파일 (process_vecs_*.py)에 대한 설명
- `is_hangul(text) -> bool`: 주어진 텍스트가 한글인지 여부를 반환하는 함수
- `load_dic(path: str) -> Set[str]`: 주어진 경로에서 사전 파일을 읽어와 집합(Set)으로 반환하는 함수, 사전에 포함된 한글 단어를 정규화하여 저장
- `blocks(files, size=65536)`: 파일을 블록 단위로 나누는 제너레이터 함수
- `count_lines(filepath)`: 주어진 파일의 총 라인 수를 세어 반환하는 함수
- 주어진 Word2Vec 모델에서 단어 벡터를 추출하고, 데이터베이스에 저장
 

---
## 의미론적 단어유사도를 활용한 카테고리/키워드 추천
저장된 단어 벡터를 활용하여 단어 간 유사도를 측정, 특정 단어와 유사한 단어들을 찾고, 해당 단어들을 기반으로 카테고리를 추천하는 기능을 수행합니다.

### 사용된 기술 및 라이브러리
- **numpy:** 다차원 배열을 처리하기 위한 라이브러리
- **pickle:** 파이썬 객체를 직렬화하고 역직렬화하는 라이브러리
- **pymysql:** MySQL 데이터베이스에 연결하고 상호작용하기 위한 라이브러리

### 사용법

1. **의존성 설치:**
   ```bash
   pip install pymysql, numpy

2. **키워드 기반 카테고리/키워드 추천, 카테고리 기반 카테고리 추천:**
   ```bash
   python process_smilar_*.py

 - [`relCategory.json`](SMLR_RECO/ByCTGY/near/relCategory.json): 카테고리 기반 추천된 관련 카테고리 정보를 JSON 형식으로 저장
- `keyword/*.dat`: 키워드 기반 추천된 관련 키워드 정보를 dat 형식으로 저장
- `category/*.json`: 키워드 기반 추천된 관련 카테고리 정보를 json 형식으로 저장
- 
### 파이썬 코드 파일 (process_smilar_*.py)에 대한 설명

- `most_similar(mat: array, idx: int, k: int) -> Tuple[array, array]`: 특정 단어에 대해 주어진 행렬에서 가장 유사한 k개의 단어와 그 유사도를 반환
- `dump_nearest(title: str, values: List[str], words: List[str], mat: array, k: int = 100) -> List[str]`: 단어의 유사도를 계산하고, 유사한 단어들을 파일로 저장, 이미 계산된 결과가 있는 경우 파일에서 로드하여 반환
- `get_nearest(title: str, values: List[str], words: List[str], mat: array) -> List[str]`: 단어의 유사도를 계산하고, 이미 계산된 결과가 있는지 확인한 후 있으면 로드하여 반환하고, 없으면 다시 계산하여 반환


---
## 라이선스
이 프로젝트는 GPL-3.0 라이선스를 따르며, 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.