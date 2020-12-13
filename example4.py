import itertools
import jieba
from tokenize_in_parallel import *

# 并行分词示例

file = "/home/zhiwen/workspace/dataset/THUCNews-title-label.txt"

def gen(file):
    with open(file, encoding="utf-8") as fp:
        text = fp.read()
    lines = text.split("\n")[:-1]
    for line in lines:
        yield line

# 分词结果按原顺序返回
tokens = tokenize_in_parallel(
    tokenize=jieba.lcut,
    generator=gen(file),
    processes=7,
    maxsize=300,
    preprocess=lambda x: x.lower()
)
print(len(tokens))
for i in range(10):
    print(tokens[i])
