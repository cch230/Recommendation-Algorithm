import json

import pymysql

with open('../DataSet/relCategory.json', 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)
print("initializing nearest words for solutions")



key_list = list(loaded_data.keys())
token_list = []

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
            SELECT DISTINCT name
            FROM tb_category
            WHERE name NOT IN ({});
       """.format(', '.join(['%s'] * len(key_list)))

        cur.execute(query, (key_list))
        for row in cur.fetchall():
            token_list.append(row[0])

print(len(key_list))
print(token_list)
