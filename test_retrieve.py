import sys
import requests

url = "http://127.0.0.1:9628/retrieve"  # 替换为实际的主机和端口


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
    # "query": "简单介绍一下中山大学信息管理学院"
    "query": "生物类在广东招生科目要求\n在广东地区，2023 年普通录取类中，生物类专业的招生科目要求是什么？"
    # "top_n": 5,
    # "method": "contriever",
    # "index": "sysu",
    # "max_tokens": 2048,
    # "refuse_threshold": 15
}


response = requests.post(url, headers = header, json=payload_faiss)
response = response.json()
t = response['results']
# print(f'response:{response}')
# print(f'res:{t}')
for i, reference in enumerate(t['contriever']):
    print('第{}条参考数据:'.format(i+1))
    print('dox_id:',reference['_id'])
    print('title:',reference['title'])
    print('content:',reference['content'])
    print('url:',reference['url'])
    print('similarity:',reference['similarity'])
    print('\n')
