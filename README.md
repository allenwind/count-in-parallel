# count-in-parallel

在数据集不大的情况下，通常我们使用类似如下代码来统计字词频率表，
```python
import itertools
import collections
X = [text1, text2, ..., textn]
words = collections.Counter(itertools.chain(*X))
```

当数据集非常大时，以上代码显得非常无力。这里提供在大数集中并行统计字词频率表的方法。

```python
import jieba

path = "THUCNews/**/*.txt"
tokens = count_in_parrallel(
    tokenize=jieba.lcut,
    batch_generator=load_batch_texts(path, limit=10000),
    processes=6,
    maxsize=300
)
```

10000个THUCNews文件，jieba作tokenize下测试：

6CPUS 8s
4CPUS 11s
2CPUs 16s
1CPUs 32s