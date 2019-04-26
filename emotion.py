#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-4-23 下午11:15
# @Author  : liyang
# @File    : emotion.py

import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from config import *


def contentHandle(content):
    content = content.replace('\ufeff', '')
    content = content.replace('\n', '')
    content = content.replace('\t', '')
    content = content.replace('\u3000', '')
    content = content.replace('\u200b', '').strip()
    return content

def entityHandle(entity):
    entity = entity.replace('\ufeff', '').replace('|', '').replace('\n', '').replace('\t', '')
    entity = entity.replace('\u3000', '').replace('。', '')
    entity = entity.replace('\u200b', '').strip()
    return entity

def extractEntity(coreEntityEmotions):
    entities = []
    for i in coreEntityEmotions:
        entity = i['entity']
        entity = entityHandle(entity)
        entities.append(entity)
    return entities

def extractEmotion(coreEntityEmotions):
    emotions = []
    for i in coreEntityEmotions:
        emotion = i['emotion']
        emotions.append(emotion)
    return emotions
'''
对原始训练数据集提取Emotion和Entity，保存成json格式
'''
def extractEntityAndEmotion():
    data = pd.read_json(TRAIN_DATA, orient='records', lines=True)
    tqdm.pandas(desc="Entity:")
    data['entity'] = data['coreEntityEmotions'].progress_apply(extractEntity)
    data['emotion'] = data['coreEntityEmotions'].progress_apply(extractEmotion)
    data.to_json(DATA_DIR+'train.json', orient='records', lines=True, force_ascii=False)

'''
读取train.json，并提取包含实体的所有句子
'''
def trainDataProcessor():
    data = pd.read_json(DATA_DIR+'train.json', orient='records', lines=True)
    tqdm.pandas(desc="emotionExamples:")
    data['emotionExamples'] = data.progress_apply(lambda x: constructSeqAndLabel(x['title'],
                                                                                x['content'],
                                                                                x['entity'],
                                                                                x['emotion']), axis=1)
    train_data = data[0: 35000]
    dev_data = data[35000: 40000]
    train_data[['newsId', 'emotionExamples']].to_json('/mnt/souhu/code/ly/data_dir/train_emotion.json',
                                                      orient='records', lines=True)
    dev_data[['newsId', 'emotionExamples']].to_json('/mnt/souhu/code/ly/data_dir/dev_emotion.json',
                                                      orient='records', lines=True)


'''
读取test.json，并提取包含实体的所有句子
'''
def testDataProcessor():
    data = pd.read_json(DATA_DIR+'test.json', orient='records', lines=True)
    tqdm.pandas(desc="emotionExamples:")
    data['emotionExamples'] = data.progress_apply(lambda x: constructSeqAndLabel(x['title'],
                                                                                x['content'],
                                                                                x['entity'],
                                                                                x['emotion']), axis=1)
    data[['newsId', 'emotionExamples']].to_json('/mnt/souhu/code/ly/data_dir/test_emotion.json',
                                                      orient='records', lines=True)

def getSeqsContainEntity(sequences, entity):
    res = []
    seq_i = 0
    while seq_i < len(sequences):
        if entity in sequences[seq_i]:
            res.append(sequences[seq_i])
        seq_i += 1
    if len(res) == 0:
        print(entity)
        print(sequences)
    return res

def constructSeqAndLabel(title, content, entities, emotions):
    content = contentHandle(content)
    # content = content.replace(' ', '')
    title = contentHandle(title)
    # title = title.replace(' ', '')
    sequences = content.split('。')
    sequences.append(title)
    seq_and_label = []
    for i in range(len(entities)):
        entity_seqs = getSeqsContainEntity(sequences, entities[i])
        seq_and_label.append([entities[i], entity_seqs, emotions[i]])
    return seq_and_label

if __name__ =='__main__':
    testDataProcessor()