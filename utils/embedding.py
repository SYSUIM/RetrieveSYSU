import logging
import sys
sys.path.append('/data2/panziyang/RetrieveSYSU')
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


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
    tokenizer = AutoTokenizer.from_pretrained(path)
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
    model = AutoModel.from_pretrained(
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

def get_reference_embedding(model, sentences):
    document_embeddings = model.get_document_embedding(sentences, bsz = 512)

    return document_embeddings


def get_query_embedding(model, sentences):
    query_embeddings = model.get_question_embedding(sentences, batch_forward = True, bsz = 512)

    return query_embeddings