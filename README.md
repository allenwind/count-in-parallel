# count-in-parallel

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

10000个THUCNews文件，jieba作tokenize下测试：

6CPUS 10000文件 8s

4CPUS 10000文件 11s

2CPUs 10000文件 16s

1CPUs 10000文件 32s

6CPUS 800000文件 350s
