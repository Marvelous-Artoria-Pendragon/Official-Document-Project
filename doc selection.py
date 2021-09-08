from gensim.models import Word2Vec
import pickle
import numpy as np
import time

model_path = 'Official Document Project/models(precise).word2vec'
model = Word2Vec.load(model_path)               # 读取词向量模型
with open('Official Document Project/docs vectors(precise).txt', 'rb') as f:
    docs_vectors = pickle.load(f)               # 读取文档向量
with open('Official Document Project/documents.txt', 'rb') as f:
        corp = pickle.load(f)                   # 读取文档内容
with open('Official Document Project/origin_corpora.txt', 'rb') as f:
    origin_corpora = pickle.load(f)
def RelatedWord(query_word = None, threshold = 0.1):
    # threshold: 关联词选择阈值参数
    # print(model.wv['办四'])
    terms = []
    if query_word == None: return
    # 选取余弦值最高的前10个词
    for key in model.wv.similar_by_word(query_word, topn = 10):
        terms.append(key)
    print(terms)
    for i in range(len(terms) - 1):
        print(terms[i])
        if (terms[i + 1][1] - terms[i][1]) > threshold:
            break
    return terms

def dos_select():
    terms = RelatedWord(query_word = '艾滋病')
    match_sim = []                  # 文档匹配相似度
    for doc_vec in docs_vectors:
        term_sim = []               # 每个拓展词和所有文档向量的相似度
        for term in terms:
            term_vector = model.wv[term[0]]         # 当前词的词向量
            num = np.dot(term_vector, doc_vec)
            denom = np.linalg.norm(term_vector) * np.linalg.norm(doc_vec)
            cos = num / denom           # 余弦值
            # sim = 0.5 + 0.5 * cos       # 归一化的相似度
            sim = cos
            term_sim.append(sim)
        match_sim.append(term_sim)

    sim_array = np.array(match_sim)                                         # 第一维是文档，第二维是拓展词
    sim_one_dim_array = sim_array.ravel()                                   # 将相似度矩阵展平
    topn = 10                                                               # 前n个查询文档
    partition = np.argpartition(sim_one_dim_array, topn)                    # 选出前n个相似度最高的文档
    num_term = len(terms)                                                   # 拓展词数量
    row = partition[:topn] // num_term; col = partition[:topn] % num_term     # 原相似度矩阵中的行列号
    print('row: ', row)
    print('col: ', col)
    end_time = time.time()
    with open('Official Document Project/query result.txt', 'w') as f:
        for i in range(topn):
            f.write(origin_corpora[row[i]] + '\n')
            # print(corp[row[i]])
    return end_time



if __name__ == '__main__':
    start_time = time.time()
    end_time = dos_select()
    print('time: ', end_time - start_time, 's')
    # print(origin_corpora[12634])
    # print(corp[12634])