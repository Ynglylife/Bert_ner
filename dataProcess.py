#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-4-14 下午8:57
# @Author  : liyang
# @File    : dataProcess.py

import pandas as pd
import time
import re
from config import *

def dataProcessor():
    def getLabel(content, entities):
        content = content.replace(' ', '')
        content = content.strip()
        content_len = len(content)
        labels = [0 for i in range(content_len)]
        for entity in entities:
            starts = [i.start() for i in re.finditer(entity, content)]
            for start in starts:
                for i in range(len(entity)):
                    labels[start+i] = 1
        line = list(zip(content, labels))
        print(line)
        return line

    data = pd.read_json(TRAIN_EXTRACTS_SEG_json % "jieba")
    data['label'] = data[0:1].apply(lambda row: getLabel(row['content'], row['entity']), axis=1)

if __name__ == '__main__':
    dataProcessor()
