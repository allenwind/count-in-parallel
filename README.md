# count-in-parallel

并行的词频统计和分词。

## 并行词频统计

在数据集不大的情况下，通常我们使用类似如下代码来统计字词频率表，

```python
import itertools
import collections
X = [text1, text2, ..., textn]
words = collections.Counter(itertools.chain(*X))
print(words.most_common(20))
```

当数据集非常大时，以上代码显得非常无力。这里提供在大数集中并行统计字词频率表的方法。

```python
import jieba

path = "THUCNews/**/*.txt"
tokens = count_in_parrallel(
    tokenize=jieba.lcut,
    batch_generator=load_batch_texts(path, limit=10000),
    processes=6,
    maxsize=300,
    preprocess=lambda x: x.lower()
)
print(len(tokens))
print(tokens.most_common(200))
print(tokens.most_common(-200))
```

## 并行分词

并行分词例子，分词结果按原句子顺序返回，这样有利于Tokenizer处理和模型训练等相关操作，

```python
import itertools
import jieba
from tokenize_in_parallel import *

file = "THUCNews-title-label.txt"

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
```


10000个THUCNews文件，jieba作tokenize下测试：

6CPUS 10000文件 8s

4CPUS 10000文件 11s

2CPUs 10000文件 16s

1CPUs 10000文件 32s

6CPUS 800000文件 350s
