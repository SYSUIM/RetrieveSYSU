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
logging.basicConfig(level=logging.DEBUG,
                    # 设置日志格式，包括时间、日志级别、消息
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    # 设置时间格式
                    datefmt='%Y-%m-%d %H:%M:%S')

local_index = {}
jsonl_file_path = "./data/sys_test/sysu_data_withid.jsonl"
index_file_path = './data/sys_test/id_index_content.pickle'
university_name = '中山大学'
match_dox_ids = []
if not os.path.exists(index_file_path):
    os.mkdir(index_file_path )
  
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


local_index =load_local_index(index_file_path)

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
    method: str = "contriever"
    index: str = "sysu"

class QueryBody(BaseModel):
    # code: int = 0
    # msg: str = "ok"
    # debug: Optional[Any] = {}
    # method: str = ""
    # references: List = []
    results: Dict = {
        # 改成名称
        "contriever": [],
    }
    success: bool

# KB = {
#     "beijing": {
#         "data": [
#             "data/beijing/knowledge.jsonl"
#         ],
#         "index": "test_beijing",
#     }
# }

KB = {
    "sysu": {
        "data": [
            "data/sys_test/sysu_data_withid.jsonl"
        ],
        "index": "sysu",
    }
}

def load_data(index):
    sentences, contents, all_search_data = [], [],{}
    for filepath in KB[index]["data"]:
        print(filepath)
        with open(filepath, "r") as f:
            for line in f.readlines():
                
                data = json.loads(line)
                dox_id = data['id']
                # if dox_id in match_dox_ids:
                # sentences.append(str(data["title"]).split('-')[-1])
                sentences.append(data["title"])
                contents.append(data)
                all_search_data[str(dox_id)] = data
    return sentences, contents, all_search_data

@app.on_event("startup")
async def startup_event():
    # local_index =load_local_index(index_file_path)
    print(f'load loacl index done')
    if not os.path.exists("index"):
        os.mkdir("index")
    vector_size = 768
    buffer_size = 50000
    model = QuestionReferenceModel('contriever/ckpt/question_encoder', 'contriever/ckpt/question_encoder')

    sentinel["model"] = model
    sentinel["index"] = {}
    sentinel["data"] = {}
    sentinel["id"] = {}

    for index_name in KB.keys():
        print("Preparing index", index_name)
        index = DenseFlatIndexer()
        print("Local Index class %s " % type(index))
        index.init_index(vector_size)

        sentences, contents, all_search_data = load_data(index_name)

        index_path = f"index/{index_name}"
        if index.index_exists(index_path):
            index.deserialize(index_path)
        else:
            document_embeddings = model.get_document_embedding(sentences)

            buffer = []
            for i, embedding in enumerate(document_embeddings):
                item = (i, embedding)
                buffer.append(item)
                if 0 < buffer_size == len(buffer):
                    index.index_data(buffer)
                    buffer = []
            index.index_data(buffer)
            print("Data indexing completed.")
            
            if not os.path.exists(index_path):
                os.mkdir(index_path)
            index.serialize(index_path)

        sentinel["index"][index_name] = index
        sentinel["data"][index_name] = contents
        # sentinel["all_data"][index_name] = all_search_data
    

        
@app.post("/retrieve", summary="retrieve")
async def retrieve(item: QueryModel) -> QueryBody:
    # return contriever_retrieve(item)
    if item.method == "contriever":
        return contriever_retrieve(item)
    elif item.method == "es":
        return es_retrieve(item)
    else:
        # return QueryBody(msg=f"Method {item.method} is not supported.")
        return QueryBody(success = False)



def contriever_retrieve(item: QueryModel) -> QueryBody:

    # item.query = str(item.query.split('\n')[1])
    
    current_year = datetime.datetime.now().year
    item.query = item.query.replace('今年',f'{current_year}年')
    item.query = item.query.replace('去年',f'{current_year-1}年')
    item.query = item.query.replace('前年',f'{current_year-2}年')
    # if '20' not in item.query:
    #     item.query = str(current_year)+'年'+item.query
    rewritten_query = item.query
    if university_name in rewritten_query:
        rewritten_query = rewritten_query.replace(university_name,'')
    time1 = time.time()
    matched_res = search(rewritten_query)
    match_time_used = time.time() - time1
    print(f'original query:{item.query}')
    logging.info(f'index search: {rewritten_query}')
    print("substring match search time: %f sec." % match_time_used)

    if len(matched_res)>1000:
        matched_res = matched_res[:1000]
    print(f'matched_res[:10]:{matched_res[:10]}\n')
    match_dox_ids = [i[0] for i in matched_res]

    question_embedding = sentinel["model"].get_question_embedding(rewritten_query)
    time0 = time.time()
    top_docs = item.top_n
    top_results_and_scores = sentinel["index"][item.index].search_knn(question_embedding, top_docs)
    time_used = time.time() - time0
    print("index search time: %f sec." % time_used)
    # print(top_results_and_scores )
    # qBody = QueryBody(method = "contriever", success = True)
    qBody = QueryBody(success = True)
    references = []

    emb_res = [(x, y) for a, b in top_results_and_scores for x, y in zip(a, b)]
    print(f'emb_res[:10]:{emb_res[:10]}\n')

    weighted_res = weighted_average(emb_res, matched_res)
    # weighted_res = weighted_res[2:]
    print(f'weighted_res[:20]:{weighted_res[:20]}')

    lst = [i[1] for i in weighted_res]

    best_id = find_max_second_order_diff(lst)
    print(f'best_id:{best_id}\n')

    weighted_res = weighted_res[:best_id]

    weighted_res = [i for i in weighted_res if i[1]>0.5]

    print(f'weighted_res result:{ weighted_res}')
    # print(f'sentinel["data"][item.index]:{len(sentinel["data"][item.index])}')

    for qid, result_score in enumerate(weighted_res):
        doc_id, score = result_score
        references.append({
            "score": float(score),
            "source": sentinel["data"][item.index][doc_id]
        })
        # print(f"Contriever score %.3f" % float(score))
        # print('title:',sentinel["data"][item.index][doc_id]["title"])
        # print('content:',sentinel["data"][item.index][doc_id]["content"])
        # print()

        reference = {  # 第一个资料片段
            "_id": doc_id,  # 这个资料的编号，便于后续复盘
            "title": sentinel["data"][item.index][doc_id]["title"],  # 也相当于QA中的Q
            "content": sentinel["data"][item.index][doc_id]["content"],  # 也相当于QA中的A
            "article_name": sentinel["data"][item.index][doc_id]["title"],  # 资料来源的文章名称（多个资料片段可以来源于同一个文章），可用于参考文献的展示，也可以留空
            "url": sentinel["data"][item.index][doc_id]["url"],  # 该文章来源的url，可以留空
            "similarity": float(score)  # query和这个资料片段的相似度
        }
        qBody.results["contriever"].append(reference)

    # return QueryBody(method="contriever", references=references)
    return qBody


def es_retrieve(item: QueryModel) -> QueryBody:
    data = {
        "query": {
            "bool": {
                "must": [
                    {"match": { "title": item.query }},
                    {"match": { "content": item.query }}
                ]
            }
        }
    }
    json_data = json.dumps(data, ensure_ascii=False, indent=4)
    proc = subprocess.check_output(['bash', "es/search.sh", KB[item.index]["index"], json_data])
    result = json.loads(proc)
    
    if result["hits"]["total"] == 0:
        return QueryBody(method="es", references=[])
    else:
        references = []
        for d in result["hits"]["hits"][:item.top_n]:
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
    build_index(jsonl_file_path, index_file_path)
    uvicorn.run(
        app="retrieve_api:app", host="127.0.0.1", port=3308, reload=True
    )
