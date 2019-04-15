#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-4-2 上午10:02
# @Author  : liyang
# @File    : createWords.py

import pandas as pd
import time
import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool
from multiprocessing import cpu_count
from config import *
from nlpUtil import NLPUtil

def extractAll():
    data = pd.read_json(TRAIN_DATA, orient='records', lines=True)
    data['entity1'] = data['coreEntityEmotions'].apply(lambda x: x[0]['entity'].strip())
    data['emotion1'] = data['coreEntityEmotions'].apply(lambda x: x[0]['emotion'].strip())

    def getEntity2(x):
        try:
            return x[1]['entity'].strip()
        except:
            return np.NaN
    data['entity2'] = data['coreEntityEmotions'].apply(getEntity2)

    def getEmotion2(x):
        try:
            return x[1]['emotion'].strip()
        except:
            return np.NaN
    data['emotion2'] = data['coreEntityEmotions'].apply(getEmotion2)

    def getEntity3(x):
        try:
            return x[2]['entity'].strip()
        except:
            return np.NaN
    data['entity3'] = data['coreEntityEmotions'].apply(getEntity3)

    def getEmotion3(x):
        try:
            return x[2]['emotion'].strip()
        except:
            return np.NaN
    data['emotion3'] = data['coreEntityEmotions'].apply(getEmotion3)
    data.to_csv(TRAIN_EXTRACTS, sep='\t', index=False, encoding='utf-8')

def getEntities():
    if not os.path.exists(TRAIN_EXTRACTS):
        extractAll()
    data = pd.read_csv(TRAIN_EXTRACTS, sep='\t', encoding='utf-8')
    dictionary = list(data['entity1']) + list(data['entity2']) + list(data['entity3'])
    dictionary = list(set(dictionary))
    dictionary.remove(np.nan)
    with open(ENTITY_WORDS, 'w') as f:
        for item in dictionary:
            f.write(item + '\n')

def segData(nthread=None):
    if nthread == None:
        nthread = cpu_count() - 3
    data = pd.read_json(TRAIN_DATA, orient='records', lines=True)
    size = data.shape[0] // nthread
    res = []
    for i in range(nthread - 1):
        sub_data = data.loc[i*size: (i+1)*size-1]
        res.append(sub_data)
    sub_data = data.loc[(nthread-1)*size:]
    res.append(sub_data)
    return res

def segContentWithMultiProc(nthread=None, seg='jieba'):
    start = time.time()
    if nthread == None:
        nthread = cpu_count() - 3
    #procs = Pool(nthread)
    procs = ProcessPoolExecutor(nthread)
    seg_data = segData(nthread)
    if seg == 'jieba':
        res = procs.map(segContentWithJieba, seg_data)
        procs.shutdown()
        end = time.time()
        print('runs %0.2f seconds.' % (end - start))
        data = pd.concat(res)
        print(data.shape)
        data.to_csv(TRAIN_EXTRACTS_SEG % seg, columns=['newsId', 'content_seg'], sep='\t', index=False,
                    encoding='utf-8')
    elif seg == 'pkuseg':
        res = procs.map(segContentWithPkuseg, seg_data)
        procs.shutdown()
        end = time.time()
        print('runs %0.2f seconds.' % (end - start))
        data = pd.concat(res)
        print(data.shape)
        data.to_csv(TRAIN_EXTRACTS_SEG % seg, columns=['newsId', 'content_seg'], sep='\t', index=False,
                    encoding='utf-8')
    elif seg == 'hanlp':
        res = procs.map(segContentWithHanLP, seg_data)
        procs.shutdown()
        end = time.time()
        print('runs %0.2f seconds.' % (end - start))
        data = pd.concat(res)
        print(data.shape)
        data.to_csv(TRAIN_EXTRACTS_SEG % seg, columns=['newsId', 'content_seg'], sep='\t', index=False,
                    encoding='utf-8')
    else:
        print("No such segment choice")
        exit(0)

def segContentWithJieba(sub_data):
    tqdm.pandas(desc="jieba Process:")
    sub_data['content_seg'] = sub_data['content'].progress_apply(NLPUtil.divideWordWithJieba)
    return sub_data

def segContentWithPkuseg(sub_data):
    tqdm.pandas(desc="pkuseg Process:")
    sub_data['content_seg'] = sub_data['content'].progress_apply(NLPUtil.divideWordWithPkuseg)
    return sub_data

def segContentWithHanLP(sub_data):
    tqdm.pandas(desc="hanlp Process:")
    sub_data['content_seg'] = sub_data['content'].progress_apply(NLPUtil.divideWordWithHanLP)
    return sub_data

if __name__ == '__main__':
    segContentWithMultiProc(nthread=20, seg='jieba')

