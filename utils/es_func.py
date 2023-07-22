import logging
from elasticsearch import Elasticsearch

# 定义索引个数，主要是为了指定"vector"为dense_vector
def create_es_index(es: Elasticsearch, index: str):
    mapping = {
        "properties": {
            "title": {
                "type": "text"
                },
            "url": {
                "type":"text"
                },
            "id": {
                "type":"long"
                },
            "content": {
                "type": "text"
                },
            "vector": {
                "type": "dense_vector",
                "dims": 768
                }
        }
    }
    # es.indices.create(index=KB["sysu"]["index"], body={"mappings": mapping})
    es.indices.create(index = index, body={"mappings": mapping})


# 将json数据导入数据库
def index_data(es: Elasticsearch, data_file_path: str, index: str):
    # data_file = KB["sysu"]["data"][0]
    lines = open(data_file_path, 'r', encoding='utf-8').readlines()
    data = [eval(line) for line in lines]
    sentences = [str(i["title"]).split('-')[-1] for i in data]

    data = data
    sentences = sentences

    all_vec = get_bert_embedding(sentences)

    logging.info(f'begin to import data to es...')
    # time1 = time.time()
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
        batch_data = bulk_data[i : i + batch_size]
        res = es.bulk(index = index, body = batch_data, refresh=True)
        print(es.count(index = index))


    # cost_time = time.time()-time1
    logging.info(f'import data to es done...')


def es_data(es: Elasticsearch, ):
    create_es_index(es = es, index = index)
    index_data(es = es, data_file_path = data_file_path, index = index)