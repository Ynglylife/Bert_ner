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
if __name__ == '__main__':
    multiProcessForTestData()
