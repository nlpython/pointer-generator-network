import pandas as pd
import json
import numpy
import pkuseg
from tqdm import tqdm

with open('train.json', 'r', encoding='utf8') as f:
    data = json.load(f)

def build_vocab():

    word_count = {}
    seg = pkuseg.pkuseg(model_name='web')
    for content in tqdm(data, desc='count word and build vocab'):

        words = seg.cut(content['content']) + seg.cut(content['title'])
        for word in words:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1

    vocab = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    with open('vocab', 'w', encoding='utf-8') as f:
        for word, count in vocab:
            f.write(word + '\n')

def describe():
    seg = pkuseg.pkuseg(model_name='web')
    title_lens = []
    content_lens = []
    for content in tqdm(data):
        title_words = seg.cut(content['title'])
        title_lens.append(len(title_words))

        content_words = seg.cut(content['content'])
        content_lens.append(len(content_words))

    df = pd.DataFrame({'title': title_lens, 'content': content_lens})
    print(df.describe())


if __name__ == '__main__':
    describe()


