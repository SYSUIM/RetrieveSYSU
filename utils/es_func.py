import sys
sys.path.append('/data2/panziyang/RetrieveSYSU')
import logging
from typing import Literal

from tqdm import tqdm
import spacy
from elasticsearch import Elasticsearch, helpers

from embedding import get_bert_embedding, get_reference_embedding, get_query_embedding
from contriever.faiss_contriever import QuestionReferenceModel


def spliter(article, method: Literal['spacy', 'langchain']):
    if method == 'spacy':
        nlp = spacy.load('/data2/panziyang/zh_core_web_sm/zh_core_web_sm-3.6.0')
        doc = nlp(article)
        sentences = [sent.text for sent in doc.sents if sent.text != '..']

    return sentences
    

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
            "article_id": {
                "type": "long"
            },
            "content": {
                "type": "text"
                },
            "title_embedding": {
                "type": "dense_vector",
                "dims": 768
                },
            "sentence_embedding": {
                "type": "dense_vector",
                "dims": 768
                },
            "sentence": {
                "type": "text"
                },
            "serial": {
                "type": "long"
                },
            "content_length": {
                "type": "long"
                },
            "sentence_length": {
                "type": "long"
                }
        }
    }

    es.indices.create(index = index, body={"mappings": mapping})


# 将json数据导入数据库
def index_data(es: Elasticsearch, data_file_path: str, index: str):
    lines = open(data_file_path, 'r').readlines()
    data = [eval(line) for line in lines]
    titles = [str(i["title"]).split('-')[-1] for i in data]
    articles = [str(i["content"]) for i in data]

    model = QuestionReferenceModel('/data2/panziyang/RetrieveSYSU/contriever/ckpt/question_encoder', '/data2/panziyang/RetrieveSYSU/contriever/ckpt/question_encoder', device = 'cuda:4')
    
    logging.info(f'begin to generate data to es...')
    bulk_data = []
    id = 0
    for article in tqdm(data[:]):
        serial = 0

        title = str(article["title"]).split('-')[-1]
        title_embedding = get_query_embedding(model, title)
        sentences = spliter(str(article["content"]), method = 'spacy')
        sentence_embedding = get_reference_embedding(model, sentences)
        
        
        for idx, embedding in enumerate(sentence_embedding):
            source = {}
            
            source["title"] = article["title"]
            source["url"] = article["url"]
            source["id"] = article["id"]
            source["article_id"] = article["id"]
            source["content"] = article["content"]
            source["title_embedding"] = title_embedding[0]
            source["sentence_embedding"] = embedding
            source["sentence"] = sentences[idx]
            source["serial"] = serial
            source["content_length"] = len(str(article["content"]))
            source["sentence_length"] = len(source["sentence"])
            
            bulk_data.append(
                {
                    '_index': index,
                    '_id': id,
                    '_source': source
                }
            )

            serial += 1
            id += 1

    logging.info(f'generation done...')
    logging.info(f'begin to import data to es...')
    helpers.bulk(es, bulk_data)
    logging.info(f'import data to es done...')


# def es_data(es: Elasticsearch, ):
#     create_es_index(es = es, index = index)
#     index_data(es = es, data_file_path = data_file_path, index = index)


if __name__ == '__main__':
    import sys
    sys.path.append('/data2/panziyang/RetrieveSYSU')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    es = Elasticsearch(['localhost:9200'], timeout=120)
    index = 'scu1'
    create_es_index(es = es, index = index)
    index_data(es, '/data2/panziyang/RetrieveSYSU/data/scu/scu_data_withid.jsonl', index = index)