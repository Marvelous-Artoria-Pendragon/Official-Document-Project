from math import log10, log2
from gensim import corpora
from gensim.models import Word2Vec
import numpy as np
import pickle

model_path = 'Official Document Project/models(precise).word2vec'
def LSF(cos, t = 30):
    return (1 - cos) * t

def test():
    with open('Official Document Project/documents.txt', 'rb') as f:
        corp = pickle.load(f)
    diction = corpora.Dictionary.load('Official Document Project/dictionary(precise).txt')
    model = Word2Vec.load(model_path)
    
    docs_vectors = []                # 所有文档向量
    theta = 0.2                      # 平滑系数
    for doc in corp:
        # tmp_vec = []                    # 存储一个csv中的文档向量
        for term in doc:
            vectors = []                                                # 原词向量
            Weights = []                                                # 一篇文档的所有词的权重
            Weighted_vectors = []                                       # 带权词向量
            # for term in doc:
            vector = model.wv[term]                                 # 词所对应的向量
            token_id = diction.token2id[term]                       # 词典中的编号
            tf = diction.cfs[token_id]                              # 词频
            N = diction.num_docs                                    # 语料库中文档总数
            n = diction.dfs[token_id]                               # 包含该词的文档数
            W = log10(theta * tf + 5) * (log2 (N / n)) ** 2         # 该词权重
            vectors.append(vector)
            Weights.append(W)
            Weighted_vectors.append(vector * W)
            
        Weighted_vectors = np.array(Weighted_vectors)
        vectors = np.array(vectors)
        c1 = np.sum(Weighted_vectors, axis = 0) / sum(Weights)                   # TF-IDF加权法
        c2 = np.sum(vectors, axis = 0) / diction.num_pos                         # 词向量平均值
        C = (c1 + c2) / 2
        docs_vectors.append(C)
    print(docs_vectors)
    with open('Official Document Project/docs vectors(precise).txt', 'wb') as f:
        pickle.dump(docs_vectors, f)              # 保存文档向量


    # print(diction.token2id['农村'])


if __name__ == '__main__':
    test()

# corp = corpora.Dictionary.load('Official Document Project/dictionary(precise).txt')
    # print('dfs:', corp.dfs)
    # word_id_dict = corp.token2id
    # print('word_id_dict: ', word_id_dict)
    # print(corp.keys)
    # print(corp.values)