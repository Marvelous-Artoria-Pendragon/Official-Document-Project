import os
from whoosh import index, sorting
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.query import *
from whoosh.qparser import MultifieldParser, OrGroup, QueryParser
from jieba.analyse import ChineseAnalyzer
from gensim.models import Word2Vec
from math import e
from dateutil import parser
import re
import csv
import time
import jieba.analyse

# 添加自定义新词
jieba.load_userdict('Official Document Project/newwords.txt')

# 停用词列表
stopwords = [line.strip() for line in open('Official Document Project/stopwords.txt', encoding = 'utf-8').readlines()] 
reg = "[^\u4e00-\u9fa5a-zA-Z0-9]"                            # 只保留中文、大小写字母\数字

class OFSearchEngine():
    def __init__(self, threshold = 0.1, t = 30, topn = 10):
        model_path = 'Official Document Project/models(search).word2vec'
        self.thershold = threshold
        self.t = t
        self.topn = topn
        self.model = Word2Vec.load(model_path)               # 读取词向量模型

    def RelatedWord(self, query_word = None, threshold = 0.1):
        # threshold: 关联词选择阈值参数
        # print(model.wv['办四'])
        terms = []
        if query_word == None: return
        # 选取余弦值最高的前10个词
        for key in self.model.wv.similar_by_word(query_word, topn = self.topn):
            terms.append(key)
        
        print(terms)
        print(terms[0])
        for i in range(1, len(terms)):
            print(terms[i])
            if (terms[i][1] - terms[i - 1][1]) > threshold:
                break
        return terms

    def createIndex(self):
        # 创建索引结构
        schema = Schema(index = ID(stored = True),
                        classification = TEXT(stored = True, analyzer = ChineseAnalyzer()),
                        release_agency = ID(stored = True),
                        context_date = DATETIME(stored = True),
                        title = TEXT(stored = True, analyzer = ChineseAnalyzer()),
                        reference_number = ID(stored = True),
                        release_date = DATETIME(stored = True, sortable = True),
                        link = ID(stored = True),
                        content = TEXT(stored = True, analyzer = ChineseAnalyzer()))

        indexdir = 'Official Document Project/indexdir/'        # 索引目录文件路径
        if not os.path.exists(indexdir):
            os.mkdir(indexdir)
        ix = create_in(indexdir, schema)                        # 增加需要建立索引的文档
        writer = ix.writer(limitmb = 256)
        for i in range(1, 9):
            print('正在读取第' + str(i) + '个文件..')
            with open ('Official Document Project/OF' + str(i) + '.csv', 'r', encoding = 'utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['正文'] == None: continue        # 检验是否读取成功
                    if row['发布日期'] == '无数据': row['发布日期'] = '1970-01-01'
                    writer.add_document(index = row['索引号'],
                                        classification = row['分类'],
                                        release_agency = row['发布机构'],
                                        context_date = parser.parse(row['成文日期']),
                                        title = row['标题'],
                                        reference_number = row['文号'],
                                        release_date = parser.parse(row['发布日期']),
                                        link = row['链接'],
                                        content = row['正文'])
        writer.commit()
        print('索引创建成功！')

    def save_result(self, results, maxdocs = 20):
        # 保存查看所有搜索到的文档
        with open('Official Document Project/query result.txt', 'w', encoding = 'utf-8', ) as f:
            for i in range(min(maxdocs, len(results))):
                f.writelines(results[i]['index'] + ',' + 
                                results[i]['classification'] + ',' + 
                                results[i]['release_agency'] + ',' + 
                                str(results[i]['context_date']) + ',' + 
                                results[i]['title'] + ',' + 
                                results[i]['reference_number'] + ',' + 
                                str(results[i]['release_date']) + ',' + 
                                results[i]['link'] + ',' + 
                                results[i]['content'] + '\n')

    def search(self, query_word):
        if query_word == None: return                               # 查询串为空
        pattern = re.compile("\d+年\d+月\d+日|\d+\D+\d+\D+\d+")      # 匹配时间
        time_result = re.findall(pattern, query_word)               # 提取的时间
        query_word = re.sub(pattern, '', query_word)
        terms = re.sub(reg, ' ', query_word)                # 去杂符号
        
        line_seg = '|'.join(jieba.cut(terms))
        tmp_list = line_seg.split('|')                      # 搜索语句分词后的列表
        tmp_list = [w for w in tmp_list if w not in stopwords and w != ' ']
        querystring = ''                # 分析后的查询语句
        for split_term in tmp_list:
            querystring += split_term + '^{} OR '.format(e)
            try:
                related_terms = self.RelatedWord(query_word = split_term)
            except KeyError:
                print('词库未找到' + split_term + '相关词!')
                continue
            for related_term in related_terms:              # 拓展词取OR连接
                querystring += related_term[0] + '^{} OR '.format(e ** related_term[1])
            querystring = querystring[:-3]                  # 原始查询词取AND连接

        ix = open_dir('Official Document Project/indexdir/')
        with ix.searcher() as searcher:
            query = MultifieldParser(['release_date', 'title', 'content'], ix.schema).parse(querystring)       # 从两个域中搜索
            print(querystring)
            release_dates = sorting.FieldFacet('release_date')
            maxdocs = 3000
            results = searcher.search(query, limit = maxdocs, sortedby = release_dates, reverse = True)     # 先按发布日期排序，再按得分排序
            print('匹配文档数目：', len(results))
            # print(results[13]['content'])     # 查看某个文档

            self.save_result(results, maxdocs)       # 保存结果
           
            
            # results = searcher.find('content', query_word)
            # print('相关文档共%d份' % len(results))
            # for i in range(min(10, len(results))):
            #     print(results[i])


if __name__ == '__main__':
    engine = OFSearchEngine()
    # engine.createIndex()
    start_time = time.time()
    engine.search('上海')
    end_time = time.time()
    print('time: ', end_time - start_time, 's')


    # query_word = '201/23/12成文日期，2021.1.19发布, 2011年10月18日起'
    # pattern = re.compile("\d+年\d+月\d+日|\d+.\d+.\d+")
    # time_result = re.findall(pattern, query_word)
    # repl_querystring = re.sub(pattern, '', query_word)
    # print(time_result)
    # print(repl_querystring)
