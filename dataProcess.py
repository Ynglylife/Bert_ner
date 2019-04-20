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
直接对文本进行分割，500个词划分一次，并且没有加标题
'''
def multiProcessForTrain(nthread = 8):
    data = pd.read_json(TRAIN_EXTRACTS_SEG_json % "jieba")
    size = data.shape[0] // nthread
    seg_data = []
    for i in range(nthread - 1):
        sub_data = data[i * size: (i + 1) * size]
        seg_data.append(sub_data)
    sub_data = data[(nthread - 1) * size:]
    seg_data.append(sub_data)

    start = time.time()
    procs = ProcessPoolExecutor(nthread)
    res = procs.map(trainDataProcessor, seg_data)
    procs.shutdown()
    end = time.time()
    print('runs %0.2f seconds.' % (end - start))
    data = pd.concat(res)
    print(data.shape)

    train_data = data[0:30000]
    test_data = data[30000:40000]
    sub_train_data = data[0:100]
    train_data[['newsId', 'label']].to_json('/mnt/souhu/code/ly/data_dir/train.json', force_ascii=False)
    test_data[['newsId', 'label']].to_json('/mnt/souhu/code/ly/data_dir/dev.json', force_ascii=False)
    sub_train_data[['newsId', 'label']].to_json('/mnt/souhu/code/ly/data_dir/sub_train.json', force_ascii=False)

def trainDataProcessor(subdata):
    tokenier = tokenization.FullTokenizer(vocab_file='data_dir/vocab.txt', do_lower_case=True)
    def getLabel(content, entities):
        content = content.replace('\ufeff', '')
        content = content.replace(' ', '')
        content = content.replace('\n', '')
        content = content.replace('\t', '')
        content = content.replace('\u3000', '')
        content = content.replace('\u200b', '').strip()
        content_tokens = tokenier.tokenize(content)
        content_len = len(content_tokens)
        labels = [0 for i in range(content_len)]
        entities_tokens = []
        for entity in entities:
            entity = entity.replace('\ufeff', '').replace('|', '')
            entity = entity.replace('*', '\*')
            entity = entity.replace('(', '\(')
            entity = entity.replace('（', '')
            entity = entity.replace(')', '\)')
            entity = entity.replace('）', '')
            entity = entity.replace('+', '\+').strip()
            entity_tokens = tokenier.tokenize(entity)
            entities_tokens.append(entity_tokens)
        index = 0
        while index < content_len:
            for tokens in entities_tokens:
                if content_tokens[index:index+len(tokens)] == tokens:
                    labels[index:index+len(tokens)] =[1]*len(tokens)
                    index += len(tokens)
                    break
            index += 1
        line = []
        index = 0
        while index+500 <= content_len:
            temp_content = content_tokens[index:index+500]
            temp_label = labels[index:index+500]
            line.append((temp_content, temp_label))
            index += 500
        if index < content_len:
            temp_content = content_tokens[index:]
            temp_label = labels[index:]
            assert len(temp_label) == len(temp_content)
            line.append((temp_content, temp_label))
        return line
    tqdm.pandas(desc="data Process:")
    subdata['label'] = subdata.progress_apply(lambda row: getLabel(row['content'], row['entity']), axis=1)
    return subdata

def handleChar():
    a = ['\ufeff', '\ufeff', '\ufeff', '\ufeff', '\ufeff',
         '更', '多', '资', '讯', '可', '登', '录', '运', '营', '商', '财', '经', '网',
         '（', 't', 'e', 'l', 'w', 'o', 'r', 'l', 'd', '.', 'c', 'o', 'm', '.', 'c', 'n', '）',
         '，', '也', '可', '关', '注', '微', '信', '公', '众', '号', 't', 'e', 'l', '_', 'w', 'o',
         'r', 'l', 'd', '\ufeff', '运', '营', '商', '财', '经', '网', '杨', '丹', '丹', '/', '文', '近', '日',
         '，', '中', '航', '三', '鑫', '股', '份', '有', '限', '公', '司', '披', '露', '了', '2', '0', '1', '9',
         '年', '第', '一', '季', '度', '业', '绩', '预', '告', '。', '预', '告', '显', '示', '，', '在', '其', '业',
         '绩', '预', '告', '期', '间', '2', '0', '1', '9', '年', '1', '月', '1', '日', '至', '2', '0', '1', '9',
         '年', '3', '月', '3', '1', '日', '内', '归', '属', '于', '上', '市', '公', '司', '股', '东', '的', '净',
         '利', '润', '亏', '损', '4', '0', '0', '0', '万', '元', '至', '2', '0', '0', '0', '万', '元', '，', '而',
         '上', '年', '同', '期', '亏', '损', '为', '：', '7', '8', '6', '.', '5', '1', '万', '元', '，', '同', '比',
         '亏', '损', '幅', '度', '较', '大', '。', '据', '悉', '，', '公', '司', '一', '季', '度', '业', '绩', '变',
         '动', '如', '此', '之', '大', '的', '主', '要', '原', '因', '是', '公', '司', '光', '伏', '产', '业', '去',
         '年', '一', '季', '度', '盈', '利', '情', '况', '较', '好', '，', '今', '年', '一', '季', '度', '受', '光',
         '伏', '“', '5', '3', '1', '”', '政', '策', '影', '响', '，', '产', '销', '量', '下', '降', '，', '经', '营',
         '亏', '损', '所', '致', '。', '\ufeff', '\ufeff', '\ufeff', '\ufeff', '\ufeff', '\ufeff', '运', '营', '商',
         '财', '经', '网', '（', '官', '方', '微', '信', '公', '众', '号', 't', 'e', 'l', '_', 'w', 'o', 'r', 'l',
         'd', '）', '—', '—', '主', '流', '财', '经', '媒', '体', '，', '一', '家', '全', '面', '覆', '盖', '科',
         '技', '、', '金', '融', '、', '证', '券', '、', '汽', '车', '、', '房', '产', '、', '食', '品', '、', '医',
         '药', '及', '其', '他', '各', '种', '消', '费', '品', '报', '道', '的', '原', '创', '资', '讯', '网', '站',
         '。'
    ]
    data = " ".join(a)
    data = data.replace('\ufeff', '').strip()
    print(data.split(" "))

def multiProcessForTestData(nthread=8):
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
    res = procs.map(testdataProcess, seg_data)
    procs.shutdown()
    end = time.time()
    print('runs %0.2f seconds.' % (end - start))
    data = pd.concat(res)
    print(data.shape)

    data[['newsId', 'label']].to_json('/mnt/souhu/code/ly/data_dir/test.json', force_ascii=False)

def testdataProcess(sub_data):
    tokenier = tokenization.FullTokenizer(vocab_file='data_dir/vocab.txt', do_lower_case=True)
    def getLabel(content):
        content = content.replace('\ufeff', '')
        content = content.replace(' ', '')
        content = content.replace('\n', '')
        content = content.replace('\t', '')
        content = content.replace('\u3000', '')
        content = content.replace('\u200b', '').strip()
        content_tokens = tokenier.tokenize(content)
        content_len = len(content_tokens)
        labels = [0 for i in range(content_len)]
        line = []
        index = 0
        while index+500 <= content_len:
            temp_content = content_tokens[index:index+500]
            temp_label = labels[index:index+500]
            line.append((temp_content, temp_label))
            index += 500
        if index < content_len:
            temp_content = content_tokens[index:]
            temp_label = labels[index:]
            assert len(temp_label) == len(temp_content)
            line.append((temp_content, temp_label))
        return line
    tqdm.pandas(desc="data Process:")
    sub_data['label'] = sub_data['content'].progress_apply(getLabel)


'''
对文本按照句子进行划分，同时加入标题
'''
tokenier = tokenization.FullTokenizer(vocab_file='data_dir/vocab.txt', do_lower_case=True)
util = NLPUtil()
def contentHandle(content):
    content = content.replace('\ufeff', '')
    content = content.replace(' ', '')
    content = content.replace('\n', '')
    content = content.replace('\t', '')
    content = content.replace('\u3000', '')
    content = content.replace('\u200b', '').strip()
    return content

def entityHandle(entity):
    entity = entity.replace('\ufeff', '').replace('|', '')
    entity = entity.replace('*', '\*')
    entity = entity.replace('(', '\(')
    entity = entity.replace('（', '')
    entity = entity.replace(')', '\)')
    entity = entity.replace('）', '')
    entity = entity.replace('+', '\+').strip()
    return entity

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
    sub_data['example'] = sub_data.progress_apply(lambda row: getLabel(row['title'], row['content'], row['entity']), axis=1)
    return sub_data

def multiProcessForTrainWithTitle(nthread = 8):
    data = pd.read_json(TRAIN_EXTRACTS_SEG_json % "jieba")
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
    train_data[['newsId', 'example']].to_json('/mnt/souhu/code/ly/data_dir/train_title.json', force_ascii=False)
    test_data[['newsId', 'example']].to_json('/mnt/souhu/code/ly/data_dir/dev_title.json', force_ascii=False)
    sub_train_data[['newsId', 'example']].to_json('/mnt/souhu/code/ly/data_dir/sub_train_title.json', force_ascii=False)

if __name__ == '__main__':
    multiProcessForTrainWithTitle()

    # a = [[[['江', '宁', '区'], ['可', '以'], ['指'], ['：'], ['.'], ['江', '宁', '区'], ['('], ['南', '京', '市'], [')'], ['，'], ['江', '苏', '省'], ['南', '京', '市'], ['下', '辖'], ['的'], ['一', '個'], ['市', '轄', '區']], [['江', '宁', '区'], ['('], ['上', '海', '市'], [')'], ['，'], ['上', '海', '市'], ['已'], ['撤', '销'], ['的'], ['一', '個'], ['市', '轄', '區']]],
    #      [['江', '苏', '省'], ['南', '京', '市'], ['江', '宁', '区']],
    #      [[['K-PRE', 'K-ORG', 'K-ORG'], ['0', '0'], ['0'], ['0'], ['0'], ['K-PRE', 'K-ORG', 'K-ORG'], ['0'], ['0', '0', '0'], ['0'], ['0'], ['K-PRE', 'K-ORG', 'K-ORG'], ['0', '0', '0'], ['0', '0'], ['0'], ['0', '0'], ['0', '0', '0']], [['K-PRE', 'K-ORG', 'K-ORG'], ['0'], ['0', '0', '0'], ['0'], ['0'], ['0', '0', '0'], ['0'], ['0', '0'], ['0'], ['0', '0'], ['0', '0', '0']]],
    #      [['K-PRE', 'K-ORG', 'K-ORG'], ['0', '0', '0'], ['K-PRE', 'K-ORG', 'K-ORG']]]