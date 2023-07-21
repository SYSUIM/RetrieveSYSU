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
import numpy as np
from collections import defaultdict
import jieba
import jieba.posseg as pseg
from collections import Counter
# from retriever.faiss_retriever import QuestionReferenceModel
# from retriever.indexer.faiss_indexers import DenseFlatIndexer
from contriever.faiss_contriever import QuestionReferenceModel
from contriever.indexer.faiss_indexers import DenseFlatIndexer
from elasticsearch import Elasticsearch
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import pickle
import logging
import os
import math
import numpy as np
from elasticsearch import helpers
# from elasticsearch_loader import Loader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class MyDataset(Dataset):
    def __init__(self,ids,length):
        self.ids = ids.to('cuda')
        self.length = length
       
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {'input_ids': self.ids['input_ids'][idx],
                'attention_mask': self.ids['attention_mask'][idx]}

def generate_ids(data_list):
    path = '/data/pzy2022/pretrained_model/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(path)
    ids = tokenizer(
        [data for data in data_list],
        return_tensors="pt",
        truncation=True,
        max_length=30, 
        padding = 'max_length', 
        add_special_tokens=False
    )
    length = len(data_list)
    return ids, length


def get_bert_embedding(sentences):

    path = '/data/pzy2022/pretrained_model/bert-base-chinese'
    model = BertModel.from_pretrained(
        path,
        torch_dtype = "auto"
        ).to('cuda')


    device_count = torch.cuda.device_count()
    if device_count > 1:
        model = torch.nn.DataParallel(model)

    ids,length =  generate_ids(sentences)

    batch_size = 1024
    dataset = MyDataset(ids,length)
    test_dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = False
        )

    logging.info(f'get embedding begin...')
    all_vec = []
    with torch.no_grad():
        for batch_id, data in enumerate(test_dataloader):
            input_keys = ('input_ids', 'attention_mask')
            input = {key: value for key, value in data.items() if key in input_keys}
            outputs = model(**input)
            last_layer_embeddings = outputs.last_hidden_state  # 获取最后一层的输出
            mask = input['attention_mask'].unsqueeze(-1).float()
            # 对每个样本的每个token的embedding乘上attention mask，即只保留非填充部分的信息
            last_layer_embeddings_masked = last_layer_embeddings * mask
            # 对每个样本的每个token的embedding在token序列长度维度上求平均
            mean_embeddings = torch.mean(last_layer_embeddings_masked, dim=1)
            # mean_embeddings = torch.mean(last_layer_embeddings_masked[:,10:,:], dim=1)
            vec = mean_embeddings.cpu().numpy().tolist()
            # vec = vec[:10]
            all_vec = all_vec + vec
            logging.info(f'batch_id:{batch_id}/{int(len(sentences)/batch_size)}')
    logging.info(f'get embedding done...')

    return all_vec



local_index = {}
jsonl_file_path = "./data/sys_test/sysu_data_withid.jsonl"
index_file_path = 'data/sys_test/id_index_content.pickle'
university_name = '中山大学'
match_doc_ids = []
es = Elasticsearch(['localhost:9200'], timeout=120)


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
            # data = json.loads(line)
            data = eval(line)
            title = data['title'] + data["content"]
            words = jieba_cut(title)
            for word in words:
                index[word].append(data['id'])

    with open(index_file_path, "wb") as f:
        pickle.dump(index, f)

    return index

build_index(jsonl_file_path, index_file_path)
print(f'build_index done')
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

    # if university_name in words:
    #     words.remove(university_name)
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
    top_n: int = 500
    # method: str = "es"
    method: str = "contriever"
    index: str = "sysu"


class ResultModel(BaseModel):
    score: float
    source: Dict[str, Any]
    vector: List[float]


class QueryBody(BaseModel):
    method: str = ""
    references: List[Dict[str, Union[float, Dict[str, Any]]]] = []
    results: Dict = {
        # 改成名称
        "contriever": [],
    }
    success: bool = True


KB = {
    "sysu": {
        "data": [
            "data/sys_test/sysu_data_withid.jsonl"
        ],
        "index": "sysu_test5",
    }
}

# 定义索引个数，主要是为了指定"vector"为dense_vector
def create_es_index():
    mapping = {
        "properties": {
            "title": {"type": "text"},
            "url":{"type":"text"},
            "id":{"type":"long"},
            "content": {"type": "text"},
            "vector": {"type": "dense_vector", "dims": 768}
        }
    }
    es.indices.create(index=KB["sysu"]["index"], body={"mappings": mapping})



# 将json数据导入数据库
def index_data():
    data_file = KB["sysu"]["data"][0]
    lines = open(data_file, 'r', encoding='utf-8').readlines()
    data = [eval(line) for line in lines]
    sentences = [str(i["title"]).split('-')[-1] for i in data]


    data = data
    sentences = sentences

    all_vec = get_bert_embedding(sentences)

    print(f'import es data begin...')
    time1 = time.time()
    bulk_data = []


    for idx, doc in enumerate(data):
        doc["vector"] = all_vec[idx]
        # print(doc["vector"])
        index_meta = {
            'index': {
                '_index': KB["sysu"]["index"],
                '_type': 'doc',
                '_id': doc['id']
            }
        }
        bulk_data.append(index_meta)
        bulk_data.append(doc)


    # res = es.bulk(index=KB["sysu"]["index"], body=bulk_data, refresh=True)

    batch_size = 20
    for i in range(0, len(bulk_data), batch_size):
        batch_data = bulk_data[i:i + batch_size]
        res = es.bulk(index=KB["sysu"]["index"], body=batch_data, refresh=True)
        print(es.count(index=KB["sysu"]["index"]))


    cost_time = time.time()-time1
    print(f'import es data done...')
    print(f'import es data cost time: {cost_time} sec.')


def es_data():
    create_es_index()
    index_data()

def load_data(index):
    sentences, contents, all_search_data = [], [],{}
    for filepath in KB[index]["data"]:
        print(filepath)
        with open(filepath, "r") as f:
            for line in f.readlines():
                
                data = json.loads(line)
                dox_id = data['id']
                # if dox_id in match_dox_ids:
                sentences.append(str(data["title"]).split('-')[-1])
                # sentences.append(data["title"])
                contents.append(data)
                all_search_data[str(dox_id)] = data
    return sentences, contents, all_search_data

@app.on_event("startup")
async def startup_event():

    # 使用es时可以直接将下面内容全部注释

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

    print('app start')

        
@app.post("/retrieve", summary="retrieve")
async def retrieve(item: QueryModel) -> QueryBody:
    if item.method == "contriever":
        return contriever_retrieve(item)
    elif item.method == "es":
        return es_retrieve(item)
    else:
        return QueryBody(success=False)


def contriever_retrieve(item: QueryModel) -> QueryBody:

    # item.query = str(item.query.split('\n')[1])
    
    current_year = datetime.datetime.now().year
    item.query = item.query.replace('今年',f'{current_year}年')
    item.query = item.query.replace('去年',f'{current_year-1}年')
    item.query = item.query.replace('前年',f'{current_year-2}年')
    # if '20' not in item.query:
    #     item.query = str(current_year)+'年'+item.query
    rewritten_query = item.query
    # if university_name in rewritten_query:
    #     rewritten_query = rewritten_query.replace(university_name,'')
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
    if len(lst)>1:
        print(f'lst:{lst}')

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
    q_vector = get_bert_embedding([str(item.query)])[0]
    print(q_vector)
    print(len(q_vector))
    # 测试标题，内容检索
    data1 = {
        "query": {
            "multi_match": {
                "query": item.query,
                "fields": ["title", "content"]
            }
        }
    }


    #测试向量检索
    data ={
        "query": {
            "script_score": {
            "query": {"match": {"title":item.query}},
            "script": {"source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {"query_vector": q_vector}}
                }
            }
        }



    print('------------------------------------',KB["sysu"]["index"])
    #测试对标题和内容进行检索（没问题
    result1 = es.search(index=KB["sysu"]["index"], body=data1)
    print(result1)
    print(f'data1 search end.--------------------------------')


    # try:
    #测试对向量检索（有问题，说解析data的参数错误，不知道错在哪了
    result = es.search(index=KB["sysu"]["index"], body=data)
    # except Exception as e:
    #     print(e.info['error'])
    
    print(result)
    
    references = []
    res = result["hits"]["hits"]

    if result["hits"]["total"]["value"] != 0: 

        lst = [i["_score"] for i in res]
        print(f'lst:{lst}')

        if len(lst)>1:    
            best_id = find_max_second_order_diff(lst)
            print(f'best_id:{best_id}\n')
            res = res[:best_id]

        for d in res:
            references.append({
                "score": d["_score"],
                "source": d["_source"],
                "vector": d["_source"]["vector"]
            })
            print(f"ES %.3f" % d["_score"])
            print(d["_source"]["title"], d["_score"])
            print(d["_source"]["content"])
            # print(d["_source"]["vector"])
    return QueryBody(method="es", references=references)

if __name__ == "__main__":
    # es_data()  # 使用es检索时可以直接注释这句代码，因为数据已经导入到数据库了
    uvicorn.run(
        app="retrieve_api:app", host="127.0.0.1", port=9633, reload=True
    )



# nohup python retrieve_api.py >> retrieve_api.log &