# Pointer-Generator-Network

论文[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)的pytorch实现。

## 1. 数据预处理

将数据处理为如下的json格式：

```
[
	{"title": "望海楼美国打“台湾牌”是危险的赌博", 
	 "content": "近期，美国国会众院通过法案，重申美国对台湾的承诺。对此，中国外交部发言人表示，有关法案严重违反一个中国原则和中美三个联合公报规定，粗暴干涉中国内政，中方对此坚决反对并已向美方提出严正交涉。\n事实上，中国高度关注美国国内..."}, 
    	{'title': ..., 'content':...}...
]
```

存入`software_data`文件夹，并命名为`train.json`，`dev.json`。

构建`vocab`，词表大小可在`processing/hyper_parm.py`中调整。

```
python precess.py
```

分词工具采用北大开源`pkuseg`，如需详细了解请点击[lancopku/pkuseg-python: pkuseg多领域中文分词工具; The pkuseg toolkit for multi-domain Chinese word segmentation (github.com)](https://github.com/lancopku/pkuseg-python)

## 2. 训练

```
python train.py
```

训练日志会存放在`/logs`，模型参数会保存至`/checkpoints`。

日志打印采用`loguru`，详细了解[Delgan/loguru: Python logging made (stupidly) simple (github.com)](https://github.com/Delgan/loguru)

## 3. 推理

```
python eval.py
```





## References

- [1704.04368.pdf (arxiv.org)](https://arxiv.org/pdf/1704.04368.pdf)

- [atulkum/pointer_summarizer: pytorch implementation of "Get To The Point: Summarization with Pointer-Generator Networks" (github.com)](https://github.com/atulkum/pointer_summarizer)
- [abisee/pointer-generator: Code for the ACL 2017 paper "Get To The Point: Summarization with Pointer-Generator Networks" (github.com)](https://github.com/abisee/pointer-generator)

