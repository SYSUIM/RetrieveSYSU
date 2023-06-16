import glob
import json
import logging
import time
from typing import List, Tuple, Dict, Iterator
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np

from .contriever import QuestionReferenceDensity_forPredict
from .indexer.faiss_indexers import DenseFlatIndexer

def read_json(filepath):
    data = json.load(open(filepath, "r"))
    sentences = list(data.keys())
    contents = list(data.values())
    return sentences, contents

class QuestionReferenceModel:
    def __init__(self, question_encoder_path, reference_encoder_path, device=None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(question_encoder_path)
        self.model = QuestionReferenceDensity_forPredict(question_encoder_path, reference_encoder_path)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if not device else device
        self.model = self.model.to(self.device).eval()

    def get_question_embedding(self, question) -> torch.Tensor:
        torch.cuda.empty_cache()
        with torch.no_grad():
            question_inputs = self.tokenizer([question], padding=True,
                                    truncation=True, return_tensors='pt')
            for key in question_inputs:
                question_inputs[key] = question_inputs[key].to(self.device)
            print(question_inputs["input_ids"].shape)
            
            question_embedding = self.model.question_encoder(**question_inputs) / 0.05
        return question_embedding.cpu().numpy()
    
    def get_document_embedding(self, sentences, bsz=32) -> torch.Tensor:
        torch.cuda.empty_cache()
        document_embeddings = []
        n = len(sentences)
        with torch.no_grad():
            for batch_start in tqdm(range(0, n, bsz), total=int(n/bsz)):
                batch = sentences[batch_start : batch_start + bsz]
                select_inputs = self.tokenizer(batch, padding=True,
                                                truncation=True, return_tensors='pt')
                for key in select_inputs:
                    select_inputs[key] = select_inputs[key].to(self.device)

                document_embedding = self.model.reference_encoder(**select_inputs)
                document_embeddings.append(document_embedding.cpu().numpy())
        document_embeddings = np.vstack(document_embeddings)
        return document_embeddings
    

def main():
    batch_size = 32
    vector_size = 768
    buffer_size = 50000
    question = "港澳台居住证"
    sentences, contents = read_json("../corpus/beijing_knowledge.json")
    scorer = QuestionReferenceModel('ckpt/question_encoder', 'ckpt/reference_encoder')

    index = DenseFlatIndexer()
    index.init_index(vector_size)

    question_embedding = scorer.get_question_embedding(question)
    print(question_embedding.shape)

    document_embeddings = scorer.get_document_embedding(sentences)
    print(document_embeddings.shape)

    buffer = []
    for i, embedding in enumerate(document_embeddings):
        item = (i, embedding)
        buffer.append(item)
        if 0 < buffer_size == len(buffer):
            index.index_data(buffer)
            buffer = []
    index.index_data(buffer)
    print("Data indexing completed.")

    time0 = time.time()
    top_docs = 5
    top_results_and_scores = index.search_knn(question_embedding, top_docs)
    print("index search time: %f sec." % (time.time() - time0))

    for qid, result_score in enumerate(top_results_and_scores):
        doc_ids, scores = result_score
        for doc_id in doc_ids:
            print(sentences[doc_id])


if __name__ == "__main__":
    main()