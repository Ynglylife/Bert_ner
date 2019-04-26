#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-4-14 下午8:57
# @Author  : liyang
# @File    : dataProcess.py

import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from bert import tokenization
from nlpUtil import NLPUtil
from config import *

'''
对文本按照句子进行划分，同时加入标题
'''
tokenier = tokenization.FullTokenizer(vocab_file='data_dir/vocab.txt', do_lower_case=True)              # Bert词划分工具
util = NLPUtil()                                                                                        # 分词工具
# 处理content字段
def contentHandle(content):
    content = content.replace('\ufeff', '')
    content = content.replace('\n', '')
    content = content.replace('\t', '')
    content = content.replace('\u3000', '')
    content = content.replace('\u200b', '').strip()
    return content
# 处理实体字段
def entityHandle(entity):
    entity = entity.replace('\ufeff', '').replace('|', '').replace('\n', '').replace('\t', '')
    entity = entity.replace('\u3000', '').replace('。', '')
    entity = entity.replace('\u200b', '').strip()
    return entity
# 找句子分词结果中实体分词的子list，进行标注
def handleLabels(tokens, entities_tokens, labels):
    index = 0
    while index < len(tokens):
        for entity in entities_tokens:
            if tokens[index:index + len(entity)] == entity:
                temp_label = [['K-PRE']]
                i = 1
                while i < len(entity):
                    temp_label.append(['K-ORG'])
                    i += 1
                labels[index: index + len(entity)] = temp_label
                index += len(entity)
        index += 1
# 处理训练数据，将标题加到每个句子后面（多进程处理调用函数）
def trainDataProcessWithTitle(sub_data):
    def getLabel(title, content, entities):
        content = contentHandle(content)                                # 去除特殊符号
        title = contentHandle(title)                                    # 去除特殊符号
        sequences = content.split('。')
        entities_seg = []
        for entity in entities:
            entity = entityHandle(entity)
            entity_seg = util.divideWordWithJieba(entity)
            entities_seg.append(entity_seg)

        sequences_tokens = []
        sequences_labels = []
        for seq in sequences:
            if seq == '':
                continue
            seq_tokens = []
            seq_seg = util.divideWordWithJieba(seq)
            seq_labels = [['0'] for i in range(len(seq_seg))]                       # 每句话的label， 每个词是['0']
            handleLabels(seq_seg, entities_seg, seq_labels)                         # entity在句子的位置，用位置标记填充
            pop_list = []
            for i, word in enumerate(seq_seg):
                word_tokens = tokenier.tokenize(word)
                if len(word_tokens) == 0:
                    pop_list.append(i)
                    continue
                seq_tokens.append(word_tokens)
                if len(word_tokens) == 1:
                    continue
                if seq_labels[i][0] == 'K-PRE' or seq_labels[i][0] == 'K-ORG':
                    seq_labels[i].extend(['K-ORG'] * (len(word_tokens) - 1))
                else:
                    seq_labels[i].extend(['0'] * (len(word_tokens) - 1))
            for i, index in enumerate(pop_list):
                seq_labels.pop(index-i)
            sequences_tokens.append(seq_tokens)
            sequences_labels.append(seq_labels)
        title_seg = util.divideWordWithJieba(title)
        title_labels = [['0'] for i in range(len(title_seg))]
        handleLabels(title_seg, entities_seg, title_labels)
        title_tokens = []
        pop_list = []
        for i, word in enumerate(title_seg):
            word_tokens = tokenier.tokenize(word)
            if len(word_tokens) == 0:
                pop_list.append(i)
                continue
            title_tokens.append(word_tokens)
            if len(word_tokens) == 1:
                continue
            if title_labels[i][0] == 'K-PRE' or title_labels[i][0] == 'K-ORG':
                title_labels[i].extend(['K-ORG'] * (len(word_tokens) - 1))
            else:
                title_labels[i].extend(['0'] * (len(word_tokens) - 1))
        for i, index in enumerate(pop_list):
            title_labels.pop(index-i)
        example = [sequences_tokens, title_tokens, sequences_labels, title_labels]
        return example

    tqdm.pandas(desc="process with title: ")
    sub_data['entityExamples'] = sub_data.progress_apply(lambda row: getLabel(row['title'], row['content'], row['entity']), axis=1)
    return sub_data

# 使用多进程处理训练数据
def multiProcessForTrainWithTitle(nthread = 8):
    data = pd.read_json(DATA_DIR+'train.json', orient='records', lines=True)
    size = data.shape[0] // nthread
    seg_data = []
    for i in range(nthread - 1):
        sub_data = data[i * size: (i + 1) * size]
        seg_data.append(sub_data)
    sub_data = data[(nthread - 1) * size:]
    seg_data.append(sub_data)

    start = time.time()
    procs = ProcessPoolExecutor(nthread)
    res = procs.map(trainDataProcessWithTitle, seg_data)
    procs.shutdown()
    end = time.time()
    print('runs %0.2f seconds.' % (end - start))
    data = pd.concat(res)
    print(data.shape)

    train_data = data[0:35000]
    test_data = data[35000:40000]
    sub_train_data = data[0:100]
    train_data[['newsId', 'entityExamples']].to_json('/mnt/souhu/code/ly/data_dir/train_title.json',
                                              orient='records', lines=True, force_ascii=False)
    test_data[['newsId', 'entityExamples']].to_json('/mnt/souhu/code/ly/data_dir/dev_title.json',
                                             orient='records', lines=True, force_ascii=False)
    sub_train_data[['newsId', 'entityExamples']].to_json('/mnt/souhu/code/ly/data_dir/sub_train_title.json',
                                                  orient='records', lines=True, force_ascii=False)

'''
处理测试集数据，多进程调用函数
'''
def testDataProcessWithTitle(sub_data):
    def getLabel(title, content):
        content = contentHandle(content)                                # 去除特殊符号
        title = contentHandle(title)                                    # 去除特殊符号
        sequences = content.split('。')

        sequences_tokens = []
        sequences_labels = []
        for seq in sequences:
            if seq == '':
                continue
            seq_tokens = []
            seq_seg = util.divideWordWithJieba(seq)
            seq_labels = [['0'] for i in range(len(seq_seg))]                       # 每句话的label， 每个词是['0']
            pop_list = []
            for i, word in enumerate(seq_seg):
                word_tokens = tokenier.tokenize(word)
                if len(word_tokens) == 0:
                    pop_list.append(i)
                    continue
                seq_tokens.append(word_tokens)
                if len(word_tokens) == 1:
                    continue
                seq_labels[i].extend(['0'] * (len(word_tokens) - 1))
            for i, index in enumerate(pop_list):
                seq_labels.pop(index-i)
            sequences_tokens.append(seq_tokens)
            sequences_labels.append(seq_labels)
        title_seg = util.divideWordWithJieba(title)
        title_labels = [['0'] for i in range(len(title_seg))]
        title_tokens = []
        pop_list = []
        for i, word in enumerate(title_seg):
            word_tokens = tokenier.tokenize(word)
            if len(word_tokens) == 0:
                pop_list.append(i)
                continue
            title_tokens.append(word_tokens)
            if len(word_tokens) == 1:
                continue
            title_labels[i].extend(['0'] * (len(word_tokens) - 1))
        for i, index in enumerate(pop_list):
            title_labels.pop(index-i)
        example = [sequences_tokens, title_tokens, sequences_labels, title_labels]
        return example

    tqdm.pandas(desc="process with title: ")
    sub_data['entityExamples'] = sub_data.progress_apply(lambda row: getLabel(row['title'], row['content']), axis=1)
    return sub_data

# 使用多进程处理测试数据
def multiProcessForTestWithTitle(nthread = 8):
    data = pd.read_json(TEST_DATA, orient='records', lines=True)
    size = data.shape[0] // nthread
    seg_data = []
    for i in range(nthread - 1):
        sub_data = data[i * size: (i + 1) * size]
        seg_data.append(sub_data)
    sub_data = data[(nthread - 1) * size:]
    seg_data.append(sub_data)

    start = time.time()
    procs = ProcessPoolExecutor(nthread)
    res = procs.map(testDataProcessWithTitle, seg_data)
    procs.shutdown()
    end = time.time()
    print('runs %0.2f seconds.' % (end - start))
    data = pd.concat(res)
    print(data.shape)
    data[['newsId', 'entityExamples']].to_json('/mnt/souhu/code/ly/data_dir/test_title.json',
                                                orient='records', lines=True, force_ascii=False)

if __name__ == '__main__':
    multiProcessForTestWithTitle()