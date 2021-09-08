from os import write
from gensim import corpora
import jieba.analyse
import csv
import re
import pickle

# stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'    # 停用标点符号
reg = "[^\u4e00-\u9fa5a-zA-Z]"                            # 只保留中文、大小写字母
def segment():
    documents = []                                      # 所有分词后的训练文档
    # 停用词列表
    stopwords = [line.strip() for line in open('Official Document Project/stopwords.txt', encoding = 'utf-8').readlines()]  
    # 添加自定义新词
    jieba.load_userdict('Official Document Project/newwords.txt')       
    with open('Official Document Project/corpus(search).txt', 'w', encoding = 'utf-8') as writefile:
        print('正在新建文件..')
        for i in range(1, 9):
            print('正在读取第' + str(i) + '个文件..')
            with open ('Official Document Project/OF' + str(i) + '.csv', 'r', encoding = 'utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    docs = row['正文']              # 提取正文内容
                    if docs == None: continue

                    docs = re.sub(reg, ' ', docs)
                    line_seg = '|'.join(jieba.cut_for_search(docs))
                    tmp_list = line_seg.split('|')
                    tmp_list = [w for w in tmp_list if w not in stopwords and w != ' ']
                    documents.append(tmp_list)
                    for word in tmp_list:           # 写入文本文件，以便查看分词效果
                        writefile.write(word + ' ')
                    writefile.write('\n')
                    # line_seg = ' '.join(jieba.cut(docs))    # 分词
                    # newdocs.append(line_seg)
                    # writefile.writelines(line_seg)

        print('写入完成')

    diction = corpora.Dictionary(documents)                     # 词典

    diction.save('Official Document Project/dictionary(search).txt')
    print('词典保存成功！')
    with open('Official Document Project/documents.txt', 'wb') as f:
        pickle.dump(documents, f)            # 保存处理后的训练文档，列表形式
        print('document保存成功！')
    # with open('Official Document Project/origin_corpora.txt', 'wb') as f:
    #     pickle.dump(origin_corpora, f)       # 保存原文档，列表形式
    #     print('origin_corpora保存成功！')
    

if __name__ == '__main__':
    segment()


# tf_idf = TfidfModel(corpus)
# corpus_tfidf = tf_idf[corpus]

# lsi = LsiModel(corpus_tfidf, id2word = diction, num_topics = 10)
# topics = lsi.show_topics(num_words = 5, log = 0)
# for tpc in topics:
#     print(tpc)

