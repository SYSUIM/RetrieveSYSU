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

from contriever.faiss_contriever import QuestionReferenceModel
from contriever.indexer.faiss_indexers import DenseFlatIndexer

logging.basicConfig(level=logging.DEBUG,
                    # 设置日志格式，包括时间、日志级别、消息
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    # 设置时间格式
                    datefmt='%Y-%m-%d %H:%M:%S')

app = FastAPI()
sentinel = {}


class QueryModel(BaseModel):
    query: str = ""
    top_n: int = 3
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
            "data/sys_test/sysu_data.jsonl"
        ],
        "index": "sysu",
    }
}

def load_data(index):
    sentences, contents = [], []
    for filepath in KB[index]["data"]:
        print(filepath)
        with open(filepath, "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                sentences.append(str(data["title"]).split('-')[-1])
                # sentences.append(data["title"])
                contents.append(data)
    return sentences, contents

@app.on_event("startup")
async def startup_event():
    if not os.path.exists("index"):
        os.mkdir("index")
    vector_size = 768
    buffer_size = 50000
    model = QuestionReferenceModel('contriever/ckpt/question_encoder', 'contriever/ckpt/reference_encoder')

    sentinel["model"] = model
    sentinel["index"] = {}
    sentinel["data"] = {}

    for index_name in KB.keys():
        print("Preparing index", index_name)
        index = DenseFlatIndexer()
        print("Local Index class %s " % type(index))
        index.init_index(vector_size)

        sentences, contents = load_data(index_name)

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
    question_embedding = sentinel["model"].get_question_embedding(item.query)
    time0 = time.time()
    top_docs = item.top_n
    top_results_and_scores = sentinel["index"][item.index].search_knn(question_embedding, top_docs)
    time_used = time.time() - time0
    logging.info(f'index search: {item.query}')
    print("index search time: %f sec." % time_used)

    # qBody = QueryBody(method = "contriever", success = True)
    qBody = QueryBody(success = True)
    references = []
    for qid, result_score in enumerate(top_results_and_scores):
        doc_ids, scores = result_score
        for doc_id, score in zip(doc_ids, scores):
            print(doc_id)
            references.append({
                "score": float(score),
                "source": sentinel["data"][item.index][doc_id]
            })
            print(f"Contriever %.3f" % float(score))
            print(sentinel["data"][item.index][doc_id]["title"], )
            print(sentinel["data"][item.index][doc_id]["content"])
            print()

            reference = {  # 第一个资料片段
                "_id": doc_id,  # 这个资料的编号，便于后续复盘
                "title": sentinel["data"][item.index][doc_id]["title"],  # 也相当于QA中的Q
                "content": sentinel["data"][item.index][doc_id]["content"],  # 也相当于QA中的A
                "article_name": sentinel["data"][item.index][doc_id]["title"],  # 资料来源的文章名称（多个资料片段可以来源于同一个文章），可用于参考文献的展示，也可以留空
                "url": "",  # 该文章来源的url，可以留空
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
    uvicorn.run(
        app="retrieve_api:app", host="0.0.0.0", port=9628, reload=True
    )