import os
import json
import time
import uvicorn
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional, List, Any, Union
import glob
import logging
import datetime
import json
import pickle
import os
import numpy as np
from collections import defaultdict
import jieba
import jieba.posseg as pseg
from collections import Counter
from contriever.faiss_contriever import QuestionReferenceModel
from contriever.indexer.faiss_indexers import DenseFlatIndexer
from elasticsearch import Elasticsearch
logging.basicConfig(level=logging.INFO,
                    # 设置日志格式，包括时间、日志级别、消息
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    # 设置时间格式
                    datefmt='%Y-%m-%d %H:%M:%S')

local_index = {}
jsonl_file_path = "./data/sys_test/sysu_data_withid.jsonl"
index_file_path = 'data/sys_test/id_index_content.pickle'
university_name = '中山大学'
match_dox_ids = []
es = Elasticsearch(['localhost:9200'])  


def jieba_cut(sentence):
    words = set(jieba.cut(sentence))
    words = [word for word in words if len(word) > 1]
    return words

def load_local_index(path):
    with open(path, "rb") as f:
        index = pickle.load(f)
    return index


def build_index(jsonl_file_path, index_file_path) -> dict:
    # 定义索引字典
    index = defaultdict(list)

    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            title = data['title'] + data["content"]
            words = jieba_cut(title)
            for word in words:
                index[word].append(data['id'])

    with open(index_file_path, "wb") as f:
        pickle.dump(index, f)

    return index

# build_index(jsonl_file_path, index_file_path)
# print(f'build_index done')
# local_index =load_local_index(index_file_path)

# 定义查询函数
def search(query) -> list:
    words = []
    all_words = jieba_cut(query)
    print(f'all_words:{all_words}')

    all_words = pseg.cut(query)
    for word, pos in all_words:
        if pos.startswith('n'):
            words.append(word)
        elif len(local_index[word]) > 50 :
            words.append(word)

    # for word in all_words:
    #     if len(local_index[word]) > 50 :
    #         words.append(word)
    #     else:
    #         all_n = pseg.cut(query)
    #         for _, pos in all_n:
    #             if pos.startswith('n'):
    #                 words.append(word)
    #                 break            

    if university_name in words:
        words.remove(university_name)
    print(words)   

    res_list = []
    cnt_dox_num = []

    rate_dic = defaultdict(list) #每个文档中包含的查询词
    for i, word in enumerate(words):
        print(f'({word},{len(local_index[word])})')
        if word in local_index:
            res_list = res_list + local_index[word]
            
            cnt_dox_num.append(len(local_index[word])+1)
            for dox_id in local_index[word]:
                rate_dic[str(dox_id)].append(i)

        else:
            cnt_dox_num.append(1)
            
    sum_num = sum(cnt_dox_num)
    rate = [sum_num/i for i in cnt_dox_num]
    sum_rate = sum(rate)
    
    norm_rate = [i/sum_rate for i in rate] 
    print(f'norm_rate:{norm_rate}')
    match_list = Counter(res_list).most_common()
    match_res = []
    word_num = len(words)
    for i in match_list:
        match_score = 0
        for j in rate_dic[str(i[0])]:
            match_score += norm_rate[j]
        match_res.append( ( int(i[0]), match_score*0.5 + (i[1]/word_num)*0.5) )


    return match_res

def weighted_average(lst1, lst2):

    # max_val = max(t[1] for t in lst1)
    # lst1_norm = [(t[0], t[1] / max_val) for t in lst1]
    dict2 = dict(lst2)
    res = [(t1[0], (t1[1] + dict2.get(t1[0], 0)) / 2) for t1 in lst1 if t1[0] in dict2]

    return sorted(res, key=lambda t: t[1], reverse=True)

def find_max_second_order_diff(lst):

    first_order_diff = [lst[i-1] - lst[i] for i in range(1, len(lst))]
    second_order_diff = [first_order_diff[i-1] - first_order_diff[i] for i in range(1, len(first_order_diff))]
    max_diff_idx = second_order_diff.index(max(second_order_diff)) + 1

    return max_diff_idx


app = FastAPI()
sentinel = {}


class QueryModel(BaseModel):
    query: str = ""
    top_n: int = 2000
    method: str = "es"
    index: str = "sysu"

class QueryBody(BaseModel):
    method: str = ""
    references: List[Dict[str, Union[float, Dict[str, str]]]] = []
    success: bool = True


KB = {
    "sysu": {
        "data": [
            "data/sys_test/sysu_data_withid.jsonl"
        ],
        "index": "sysu",
    }
}


def es_data():
    time1 = time.time()
    data_file = KB["sysu"]["data"][0]

    lines = open(data_file, 'r', encoding='utf-8').readlines()
    data = [eval(line) for line in lines]

    bulk_data = []
    for doc in data:
        index_meta = {
            'index': {
                '_index': KB["sysu"]["index"],
                '_type': 'doc',
                '_id': doc['id']
            }
        }
        bulk_data.append(index_meta)
        bulk_data.append(doc)

    res = es.bulk(index=KB["sysu"]["index"], body=bulk_data, refresh=True)
    cost_time = time.time()-time1
    print(f'import es data done...')
    print(f'import es data cost time: {cost_time} sec.')


@app.on_event("startup")
async def startup_event():
    es_data()
    print('app start')
    

        
@app.post("/retrieve", summary="retrieve")
async def retrieve(item: QueryModel) -> QueryBody:
    # return contriever_retrieve(item)
    if item.method == "contriever":
        # return contriever_retrieve(item)
        return 
    elif item.method == "es":
        return es_retrieve(item)
    else:
        return QueryBody(success = False)



def es_retrieve(item: QueryModel) -> QueryBody:

    print(f'query:{item.query}')

    data = {
        "query": {
            "multi_match": {
                "query": item.query,
                "fields": ["title", "content"]
            }
        }
    }

    result = es.search(index=KB["sysu"]["index"], body=data)
    references = []
    res = result["hits"]["hits"]

    if result["hits"]["total"] != 0: 

        lst = [i["_score"] for i in res]
        print(f'lst:{lst}')

        if len(lst)>1:    
            best_id = find_max_second_order_diff(lst)
            print(f'best_id:{best_id}\n')
            res = res[:best_id]

        for d in res:
            references.append({
                "score": d["_score"],
                "source": d["_source"]
            })
            print(f"ES %.3f" % d["_score"])
            print(d["_source"]["title"], d["_score"])
            print(d["_source"]["content"])
            print()
    return QueryBody(method="es", references=references)


if __name__ == "__main__":

    uvicorn.run(
        app="retrieve_api:app", host="127.0.0.1", port=9628, reload=True
    )

# nohup python retrieve_api.py >>retrieve_api.log &
