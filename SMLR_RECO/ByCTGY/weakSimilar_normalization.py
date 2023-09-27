import json


def remove_middle_elements(lst):
    length = len(lst)
    to_remove = length - 19  # 제거해야 할 요소의 수 계산

    if length % 2 == 1:  # 홀수일 경우
        mid_index = length // 2
        del lst[mid_index - to_remove // 2 : mid_index + to_remove // 2 + 1]

    else:  # 짝수일 경우
        mid_right = length // 2
        mid_left = mid_right - 1
        del lst[mid_left - (to_remove // 2 - 1) : mid_right + to_remove // 2 + 1]

def swap_front_and_back(lst):
    length = len(lst)
    mid_index = length // 2

    if length % 2 == 0:  # 리스트의 길이가 짝수일 때
        lst[:mid_index], lst[mid_index:] = lst[mid_index:], lst[:mid_index]
    else:  # 리스트의 길이가 홀수일 때
        lst[:mid_index], lst[mid_index + 1:] = lst[mid_index + 1:], lst[:mid_index]
loaded_data = {}
with open('../DataSet/relCategory.json', 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)

# newRelCategory_weak = {}
# for key, values in loaded_data.items():
#     swap_front_and_back(values)
#     if len(values) > 19:
#         # 리스트의 요소가 20개 초과일 때 가운데 요소 제거
#         remove_middle_elements(values)
#     values.append("weak")
#
# # JSON 파일로 저장
# with open('../DataSet/newRelCategory_weak.json', 'w', encoding='utf-8') as json_file:
#
print(len(loaded_data))

