#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-4-14 下午8:57
# @Author  : liyang
# @File    : dataProcess.py

import pandas as pd
import re
from tqdm import tqdm
from config import *


def dataProcessor():
    entities_exception = []
    def getLabel(content, entities):
        content = content.strip()
        content = content.replace(' ', '')
        content = content.replace('\n', '')
        content = content.replace('\t', '')
        content_len = len(content)
        labels = [0 for i in range(content_len)]
        for entity in entities:
            entity = entity.replace('|', '')
            entity = entity.replace('*', '\*')
            entity = entity.replace('(', '\(')
            entity = entity.replace('（', '')
            entity = entity.replace(')', '\)')
            entity = entity.replace('）', '')
            entity = entity.replace('+', '\+')
            starts = [i.start() for i in re.finditer(entity, content)]
            for start in starts:
                for i in range(len(entity)):
                    labels[start + i] = 1
        line = []
        index = 0
        while index+500 <= content_len:
            temp_content = content[index:index+500]
            temp_label = labels[index:index+500]
            line.append(list(zip(temp_content, temp_label)))
            index += 500
        if index < content_len:
            temp_content = content[index:]
            temp_label = labels[index:]
            assert len(temp_label) == len(temp_content)
            line.append(list(zip(temp_content, temp_label)))
        return line
    tqdm.pandas(desc="data Process:")
    data = pd.read_json(TRAIN_EXTRACTS_SEG_json % "jieba")
    data['label'] = data.progress_apply(lambda row: getLabel(row['content'], row['entity']), axis=1)
    train_data = data[0:30000]
    test_data = data[30000:40000]
    sub_train_data = data[0:100]
    train_data[['newsId', 'label']].to_json(TRAIN_EXTRACTS_BERT % "train")
    test_data[['newsId', 'label']].to_json(TRAIN_EXTRACTS_BERT % "test")
    sub_train_data[['newsId', 'label']].to_json(TRAIN_EXTRACTS_BERT % "sub_train")

if __name__ == '__main__':
    dataProcessor()
