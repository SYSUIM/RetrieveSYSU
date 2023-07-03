import json
import pickle
import os
from collections import defaultdict
import jieba
from collections import Counter
import pandas as pd

def jieba_cut(sentence):
    words = set(jieba.cut(sentence))
    words = [word for word in words if len(word) > 1]
    return words

def build_index(jsonl_file_path, index_file_path) -> dict:
    # 定义索引字典
    index = defaultdict(list)

    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            title = data["content"]
            words = jieba_cut(title)
            for word in words:
                index[word].append(data['id'])

    with open(index_file_path, "wb") as f:
        pickle.dump(index, f)

    return index


# 定义查询函数,返回为加上匹配率的匹配数据集列表,数据集保存在data_for_query_path路径
def search(query, jsonl_file_path, index_file_path, data_for_query_path) -> list:

    with open(index_file_path, "rb") as f:
        index = pickle.load(f)

    all_dic = {}
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            all_dic[str(data['id'])] = data

    words = jieba_cut(query)
    # print(words)
    res_list = []
    for word in words:
        res_list = res_list + index[word]

    count_dict = Counter(res_list)
    word_num = len(words)

    match_data = []
    for id, match_num in count_dict.items():
        dic = all_dic[id]
        dic['match'] = match_num / word_num
        match_data.append(dic)
    sorted_match_data = sorted(match_data, key=lambda x: x['match'], reverse=True)
    
    df = pd.DataFrame(sorted_match_data)
    res = df[df['match'] > 0.3]
    res.to_json(data_for_query_path, orient="records", lines=True, force_ascii=False)
    return res


if __name__ == "__main__":
    
    index_file_path = "/data2/fkj2023/projects/RetrieveSYSU/data/sys_test/id_index.pickle"
    jsonl_file_path = "/data2/fkj2023/projects/RetrieveSYSU/data/sys_test/sysu_data_withid.jsonl"
    index = build_index(jsonl_file_path, index_file_path)

    data_for_query_path = "/data2/fkj2023/projects/RetrieveSYSU/data/sys_test/data_for_query.jsonl"
    match_data = search('中山大学信息管理学院2022年录取分数线',jsonl_file_path, index_file_path, data_for_query_path)
    for _, i in match_data.iterrows():
        print(i['title'],i['match'])
