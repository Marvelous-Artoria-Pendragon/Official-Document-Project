from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary
import pickle

model_path = 'Official Document Project/models(search).word2vec'

def train():
    with open('Official Document Project/documents.txt', 'rb') as f:
        corp = pickle.load(f)
    #通过Word2vec进行训练
    

    # 读取已分词的文档训练
    # corp = []
    # with open('Official Document Project/text.txt', 'r', encoding = 'utf-8') as f:
    #     for line in f.readlines():
    #         ss = line.replace('\n', '').split(' ')
    #         corp.append(ss)

    model = Word2Vec(sg=1, vector_size = 400, window = 6, hs = 1, min_count= 0, batch_words = 10000)
    model.build_vocab(corp)
    model.train(corp, total_examples = model.corpus_count, epochs = model.epochs)
    #保存训练好的模型
    model.save(model_path)
    print('训练完成')
    


if __name__ == '__main__':
    train()
    