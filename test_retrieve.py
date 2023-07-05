import sys
import requests

url = "http://0.0.0.0:9628/retrieve"  # 替换为实际的主机和端口

# 可以不改
header = {
    "Content-Type": "application/json",  # body 类型
    "Authorization": "Bearer [token]"  # token鉴权
}

# faiss 测试
payload_faiss = {
    # "authorization":"sysu",
    # 多种输入方式
    # "query": input("请输入查询内容: "),
    # "query": sys.argv[1],
    "query": "我高考结束了，想报考中山大学，招生咨询联系方式是多少?",
    # "top_n": 5,
    # "method": "contriever",
    # "index": "sysu",
    # "max_tokens": 2048,
    # "refuse_threshold": 15
}

response = requests.post(url, headers = header, json=payload_faiss)
print(response.text)
print(response.status_code)
response = response.json()
for i, reference in enumerate(response['references']):
    print('第{}条参考数据是{}'.format(i, reference))
    print('\n')