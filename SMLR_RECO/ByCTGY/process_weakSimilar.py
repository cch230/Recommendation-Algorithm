import json
import operator
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
            if float(d) > 0.25:
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

weakWord = [
    '면세점', '축산물', '농산물', '밀키트', '마라톤용품', '관상어용품', '블루레이', '코디세트', '카디건', '청바지', '보정속옷', '시즌성내의', '부티', '워커', '웰트화',
    '신발끈', '숄더백', '힙색', '네임태그', '선캡', '페도라', '스냅백', '베레모', '가발', '커프스', '넥케이프', '부채', '시계수리용품', '골드바', '애프터선', '프라이머',
    '파운데이션', '립스틱', '매니큐어', '아로마테라피', '데오드란트', '헤어스프레이', '타투', '올인원', '전자사전', '전자책', '보이스레코더', '브랜드PC', '서버/워크스테이션',
    '마우스패드', '사운드카드', 'PC마이크', '웹캠', 'ODD', '그래픽카드', '메인보드', '파워서플라이', '선택기', '안테나', 'KVM스위치', '운영체제', '유틸리티', '자급제폰',
    '짐벌', '메모리카드', '망원경', '천체망원경', '현미경', '데크', '턴테이블', '노래반주기', '다리미', '무전기', '재봉틀', '핸드드라이어', '기타이미용가전', '드라이어',
    '에어브러시', '가습기', '라디에이터', '컨벡터', '전기쿠커', '하이라이트', '분쇄기', '하이브리드', '하이패스/GPS', '카팩', '매트리스', '거울', '나비장', '코너장',
    '책꽂이', '정자', '손잡이', '패널', '접착제/보수용품', '벽지', '우체통', '디자인문패', '화병', '오르골', '스프레드', '한실예단세트', '대자리', '버티컬', '캐노피',
    '대나무발', '자바라', '뜨개질', '보석십자수', '노리개젖꼭지', '절충형/디럭스형', '쌍둥이용', '일체형', '분리형', '슬링', '키재기', '젖병솔', '젖병집게', '디딤대',
    '돌잔치용품', '우주복', '상하세트', '공주드레스', '타이즈', '점퍼루', '물놀이용품', '봉제인형', '꿀', '식이섬유', '콜라겐', '카테킨', '히알루론산', '시네트롤', '시서스',
    '핫도그', '햄버거', '샐러드', '도시락', '누룽지', '돼지고기', '쇠고기', '닭고기', '장아찌', '깍두기', '동치미', '열무김치', '코코아', '젤리', '캐러멜', '강정', '껌',
    '한과', '화과자', '연유', '해산물/어패류', '건어물', '밀가루', '오트밀', '콩가루', '마요네즈', '고추장', '청국장', '딸기잼', '땅콩잼', '베이킹파우더', '고춧가루',
    '후추', '골뱅이/번데기', '전통주', '랜턴', '취사용품', '아이스박스', '로스트볼', '러닝머신', '일립티컬', '스텝퍼', '트위스트', '트램펄린', '스네이크보드', '글러브', '헬멧',
    '셔틀콕', '스트링', '기어백', '리트렉터', '옥토퍼스', '호스', '타격대', '샌드백', '마우스피스', '격파용품', '스코어보드/작전판', '낚싯대', '물감/포스터컬러',
    '스케치/드로잉용품', '스노우체인', '세차용품', '모형/프라모델/피규어', '서바이벌', '화폐', '우표/씰', '코스튬플레이', '아이돌굿즈', '수족관/어항', '수초', '모터',
    '여과기/여과제', '뮤직비디오', '드라마', '도마', '프라이팬', '주전자/티포트', '와인용품', '제수용품', '빨랫줄', '빨래판', '만보계', '체온계', '청진기', '파라핀용품',
    '석션기/네블라이저', '휠체어', '장례용품/수의', '돋보기', '보청기', '적외선조사기', '좌욕기', '좌훈기', '마네킹', '회전청소기', '식용식물', '숯부작', '보존화', '금꽃',
    '종이꽃', '테라리엄', '물조리개', '수반', '자갈/모래/흙', '햄스터용품', '고슴도치용품', '조류용품', '건반악기', '국악기', '기타악기', '피아노', '제도용품', '데스크용품',
    '용지', '파일/바인더', '지도/지구본', '체결용품', '운반용품', '액티비티', '와이파이/USIM', 'e쿠폰', '온라인 콘텐츠', '피트니스/PT', '크리스탈', '기호식품', '오디오북',
    '명언/잠언록', '심리', '언어학/기호학', '종교학/신화학', '자녀교육', '재테크/투자', 'CEO/비즈니스맨', '시간관리', '인간관계', '대화/협상', '여성학', '교육학', '세계사',
    '서양사', '동양사', '한국사', '한자사전', '물리학', '생물학', '천문/지구과학', '공무원', '고등고시/전문직', '편입/대학원', '교원임용고시', 'OS/데이터베이스', '웹사이트',
    '순정만화', 'SF/판타지', '학원만화', '스페인', '일본도서', '독일', '프랑스'
]

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
            if key in weakWord:
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
                               WHERE name REGEXP %s AND name NOT IN ({});
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
with open('../DataSet/relCategory_weak.json', 'w', encoding='utf-8') as json_file:
    json.dump(result_dict, json_file, ensure_ascii=False, indent=4)
