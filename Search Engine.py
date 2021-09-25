import os
from whoosh import index, sorting
from whoosh import qparser
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
import random as rd
import pickle


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
        self.model = Word2Vec.load(model_path)                       # 读取词向量模型
        self.indexdir = 'Official Document Project/indexdir/'        # 索引目录文件路径
        self.schema = Schema(index = ID(stored = True),
                             classification = KEYWORD(stored = True, analyzer = ChineseAnalyzer(), scorable = True),
                             species = KEYWORD(stored = True, scorable = True),
                             province = KEYWORD(stored = True, scorable = True),
                             release_agency = ID(stored = True),
                             context_date = DATETIME(stored = True),
                             title = TEXT(stored = True, analyzer = ChineseAnalyzer()),
                             reference_number = ID(stored = True),
                             release_date = DATETIME(stored = True, sortable = True),
                             link = ID(stored = True),
                             content = TEXT(stored = True, analyzer = ChineseAnalyzer()),
                             clickrate = NUMERIC(stored = True, sortable = True))

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

    def createHierarchicalIndex(self):
        # 创建分级索引
        if not os.path.exists(self.indexdir):
            os.mkdir(self.indexdir)
        classification = {'国务院组织机构':[], '综合政务': [], '国民经济管理、国有资产监管': [],
                          '财政、金融、审计': [], '国土资源、能源': [], '农业、林业、水利': [],
                          '工业、交通': [], '商贸、海关、旅游': [], '市场监管、安全生产监管': [],
                          '城乡建设、环境保护': [], '科技、教育': [], '文化、广电、新闻出版': [],
                          '卫生、体育': [], '人口与计划生育、妇女儿童工作': [], '劳动、认识、监察': [],
                          '公安、安全、司法': [], '民政、扶贫、救灾': [], '民族、宗教': [],
                          '对外事务': [], '港澳台侨工作': [], '国防': [], '其他': []}             # 分类索引
        for i in range(2, 9):
            print('正在读取第' + str(i) + '个文件..')
            with open ('Official Document Project/OOF' + str(i) + '.csv', 'r', encoding = 'utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['正文'] == None: continue        # 检验是否读取成功
                    if row['发布日期'] == '无数据': row['发布日期'] = '1970-01-01'
                    pattern = re.compile('[?、;"]')
                    curr_class = re.split(pattern, row['分类'])
                    isOther = True                           # 其它类标志
                    for key in classification.keys():
                        for cla in curr_class:
                            if cla in key:
                                classification[key].append(row)
                                isOther = False; break
                    if isOther == True:
                        classification['其他'].append(row)



        print('共' + str(len(classification)) + '个类别')
        for key in classification.keys():
            print('正在创建' + key + '索引...')
            ix = create_in(self.indexdir, self.schema, indexname = key)
            writer = ix.writer()
            for doc in classification[key]:
                writer.add_document(index = doc['索引号'],
                                    classification = doc['分类'],
                                    species = doc['文种'],
                                    province = doc['省份'],
                                    release_agency = doc['发布机构'],
                                    context_date = parser.parse(doc['成文日期']),
                                    title = doc['标题'],
                                    reference_number = doc['文号'],
                                    release_date = parser.parse(doc['发布日期']),
                                    link = doc['链接'],
                                    content = doc['正文'],
                                    clickrate = doc['点击量'])
            writer.commit()
        print('索引创建成功！')

    def createProvinceIndex(self):
        # 创建省份索引
        for i in range(2, 9):
            print('正在读取第' + str(i) + '个文件..')
            with open ('Official Document Project/OOF' + str(i) + '.csv', 'r', encoding = 'utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                provinces = ['广东', '国务院', '河北', '黑龙江', '江苏', '上海', '深圳']
                print('正在创建' + provinces[i - 2] + '索引...')
                ix = create_in(self.indexdir, self.schema, indexname = provinces[i - 2])
                writer = ix.writer()
                for row in reader:
                    writer.add_document(index = row['索引号'],
                                        classification = row['分类'],
                                        species = row['文种'],
                                        province = row['省份'],
                                        release_agency = row['发布机构'],
                                        context_date = parser.parse(row['成文日期']),
                                        title = row['标题'],
                                        reference_number = row['文号'],
                                        release_date = parser.parse(row['发布日期']),
                                        link = row['链接'],
                                        content = row['正文'],
                                        clickrate = row['点击量'])
                writer.commit()
        print('索引创建成功！')

    def createGeneralIndex(self):
        # 创建总索引结构
        if not os.path.exists(self.indexdir):
            os.mkdir(self.indexdir)
        ix = create_in(self.indexdir, self.schema, indexname = 'General Index')                        # 增加需要建立索引的文档
        writer = ix.writer(limitmb = 256)
        for i in range(2, 9):
            print('正在读取第' + str(i) + '个文件..')
            with open ('Official Document Project/OOF' + str(i) + '.csv', 'r', encoding = 'utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['正文'] == None: continue        # 检验是否读取成功
                    if row['发布日期'] == '无数据': row['发布日期'] = '1970-01-01'
                    writer.add_document(index = row['索引号'],
                                        classification = row['分类'],
                                        species = row['文种'],
                                        province = row['省份'],
                                        release_agency = row['发布机构'],
                                        context_date = parser.parse(row['成文日期']),
                                        title = row['标题'],
                                        reference_number = row['文号'],
                                        release_date = parser.parse(row['发布日期']),
                                        link = row['链接'],
                                        content = row['正文'],
                                        clickrate = row['点击量'])
        writer.commit()
        print('索引创建成功！')

    def save_result(self, results, maxdocs = 20):
        '''
        results: whoosh搜索结果
        maxdocs: 最大搜索结果
        '''
        # 保存查看所有搜索到的文档
        with open('Official Document Project/query result.txt', 'w', encoding = 'utf-8') as f:
            for i in range(min(maxdocs, len(results))):
                f.writelines(results[i]['index'] + ',' + 
                                results[i]['classification'] + ',' + 
                                results[i]['species'] + ',' + 
                                results[i]['release_agency'] + ',' + 
                                results[i]['province'] + ',' + 
                                str(results[i]['context_date']) + ',' + 
                                results[i]['title'] + ',' + 
                                results[i]['reference_number'] + ',' + 
                                str(results[i]['release_date']) + ',' + 
                                results[i]['link'] + ',' + 
                                results[i]['content'] + 
                                results[i]['clickrate'] + '\n')

    def search(self, query_word, province = 'General Index', cla_name = None, species = None, maxdocs = 20):
        '''
        query_word: 用户查询语句
        cla_name: 指定搜索类别
        province: 指定搜索省份, 默认全国
        species: 指定搜索文种
        maxdocs: 搜索最大文档数
        '''
        
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

        ix = open_dir(self.indexdir, indexname = province)
        with ix.searcher() as searcher:
            filter = None
            if species != None:         # 筛选文种
                filter_query = qparser.QueryParser('species', ix.schema).parse(species)
                filter = searcher.search(filter_query, limit = None)
            if cla_name != None:        # 筛选类别
                filter_query = qparser.QueryParser('classification', ix.schema).parse(cla_name)
                filter = searcher.search(filter_query, limit = None, filter = filter)

            query = MultifieldParser(['title', 'content'], ix.schema).parse(querystring)       # 用户查询语句分析，从两个域中搜索
            print(querystring)
            release_dates = sorting.FieldFacet('release_date')
            # results = searcher.search(query, limit = maxdocs, sortedby = release_dates, reverse = True)     # 先按发布日期排序，再按得分排序
            results = searcher.search(query, limit = maxdocs, sortedby = [release_dates, 'clickrate'], reverse = True, filter = filter)     # 先按发布日期排序
            print('匹配文档数目：', len(results))
            self.save_result(results, maxdocs)       # 保存结果
        
    def createfilter(self):
        index_dir = 'Official Document Project/indexdir/'           # 索引目录路径
        classification = {'国务院组织机构':[], '综合政务': [], '国民经济管理、国有资产监管': [],
                          '财政、金融、审计': [], '国土资源、能源': [], '农业、林业、水利': [],
                          '工业、交通': [], '商贸、海关、旅游': [], '市场监管、安全生产监管': [],
                          '城乡建设、环境保护': [], '科技、教育': [], '文化、广电、新闻出版': [],
                          '卫生、体育': [], '人口与计划生育、妇女儿童工作': [], '劳动、认识、监察': [],
                          '公安、安全、司法': [], '民政、扶贫、救灾': [], '民族、宗教': [],
                          '对外事务': [], '港澳台侨工作': [], '国防': [], '其他': []}             # 分类索引
        for cla in classification:      # 分类
            ix = open_dir(index_dir, indexname = cla)
            query = MultifieldParser(['classification'], ix.schema).parse(cla)
            with ix.searcher() as searcher:
                results = searcher.search(query, limit = None)
            with open('Official Document Project/filter/' + cla + '.dat', 'wb') as f:
                pickle.dump(results, f)
        

            # results = searcher.find('content', query_word)
            # print('相关文档共%d份' % len(results))
            # for i in range(min(10, len(results))):
            #     print(results[i])

    def cyclicquery(self, query_word, t, name):
        '''关联词查询'''
        term = query_word         # 最不相关关联词
        cla_name = 'General Index'
        index_dir = 'Official Document Project/indexdir/'           # 索引目录路径
        ix = open_dir(index_dir, indexname = cla_name)
        release_dates = sorting.FieldFacet('release_date')          # 按发布时间排序
        maxdocs = 20                                # 最大文档数
        with open('Official Document Project/related word/' + name + '.txt', 'w', encoding = 'utf-8') as f:                # 保存相关词
            with ix.searcher() as searcher:
                for _ in range(1, t + 1):
                    print("第" + str(_) + "次:")
                    querystring = ''                # 分析后的查询语句
                    querystring += term + '^{} OR '.format(e)
                    print(term)
                    try:
                        related_terms = self.RelatedWord(query_word = term)
                    except KeyError:
                        print('词库未找到' + term + '相关词!')
                        exit(1)
                
                    f.writelines(str(_) + '.' +  term + '\n')
                    for j in range(len(related_terms) - 1):
                        querystring += related_terms[j][0] + '^{} OR '.format(e ** related_terms[j][1])
                        f.writelines(related_terms[j][0] + '\n')        # 记录每个关联词
                    f.write('\n')
                    querystring = querystring[:-3]                  # 原始查询词取AND连接
                    
                    # query = MultifieldParser(['title', 'content'], ix.schema).parse(querystring)
                    # results = searcher.search(query, limit = maxdocs, sortedby = release_dates, reverse = True)     # 先按发布日期排序，再按得分排序
                    # print('匹配文档数目：', len(results))

                    term = related_terms[rd.randint(0, 5)][0]
                


if __name__ == '__main__':
    engine = OFSearchEngine()
    # engine.createfilter()
    # engine.createHierarchicalIndex()
    # engine.createProvinceIndex()
    start_time = time.time()
    # engine.createGeneralIndex()
    engine.search('引领世界潮流', province = '江苏')
    # L = ['法律', '民生', '农业', '林业', '畜牧业', '改革', '农村', '医疗', '金融', '财政', '科技', '教育', '水利', '民族', '旅游', '海关',
    #      '人口', '国防', '劳动', '商贸', '救灾', '卫生', '市场', '审计', '资源', '公安', '司法', '环境', '城市', '文化', '宗教', '扶贫',
    #      '新闻', '民族']
    # for word in L:
    #     engine.cyclicquery(word, 100, word)
    end_time = time.time()
    print('time: ', end_time - start_time, 's')


    # query_word = '201/23/12成文日期，2021.1.19发布, 2011年10月18日起'
    # pattern = re.compile("\d+年\d+月\d+日|\d+.\d+.\d+")
    # time_result = re.findall(pattern, query_word)
    # repl_querystring = re.sub(pattern, '', query_word)
    # print(time_result)
    # print(repl_querystring)
