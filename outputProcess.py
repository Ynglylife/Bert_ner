#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-4-23 上午9:40
# @Author  : liyang
# @File    : outputProcess.py
import pandas as pd
from tqdm import tqdm

# 在预测结果中寻找4所在位置。
def getFourIndex(arr):
    indexs = []
    for i in range(len(arr)):
        if int(arr[i]) == 4:
            indexs.append(i)
    return indexs

# 将分词结果转化为1维，直接拉成一维
def convertTo1D(tokens):
    res = []
    for token in tokens:
        res.extend(token)
    return res

# 将预测结果转化为字符（存在'[UNK]'需要处理）
def getResultForEntity():
    res = []
    with open('data_dir/model/test_results.tsv', 'r') as f:
        for line in f.read().strip().split('\n'):
            example_out = line.strip().split('\t')
            res.append(example_out)
    print("results文件读取完成")
    print(res[0])
    tqdm.pandas(desc="getLabel:")
    test_data = pd.read_json('data_dir/test_title.json', orient='records', lines=True)
    print("json文件读取完成")
    now_index = 0
    def getLabel(example):
        nonlocal now_index
        sequences_tokens = example[0]
        title_tokens = convertTo1D(example[1])

        seq_cnt = len(sequences_tokens)
        keywords = list()
        for seq_index in range(seq_cnt):
            seq_tokens = convertTo1D(sequences_tokens[seq_index])
            seq_res = res[now_index+seq_index]
            index4 = getFourIndex(seq_res)
            seq_labels = seq_res[1: index4[0]]
            title_labels = seq_res[index4[0]+1: index4[-1]]
            i = 0
            temp_str = ''
            while i < len(seq_labels):
                if seq_labels[i] == '1':
                    temp_str += seq_tokens[i].replace('##', '')
                elif seq_labels[i] == '2':
                    if temp_str != '':
                        temp_str += seq_tokens[i].replace('##', '')
                else:
                    if temp_str != '':
                        keywords.append(temp_str)
                    temp_str = ''
                i += 1

            i = 0
            temp_str = ''
            assert len(title_tokens) >= len(title_labels)
            while i < len(title_labels):
                if title_labels[i] == '1':
                    temp_str += title_tokens[i].replace('##', '')
                elif title_labels[i] == '2':
                    if temp_str != '':
                        temp_str += title_tokens[i].replace('##', '')
                else:
                    if temp_str != '':
                        keywords.append(temp_str)
                    temp_str = ''
                i += 1
        if "" in keywords:
            keywords.remove('')
        keywords = list(set(keywords))
        level = set()
        for i in range(len(keywords)):
            for j in range(i+1, len(keywords)):
                if keywords[i] in keywords[j]:
                    level.add(keywords[i])
                if keywords[j] in keywords[i]:
                    level.add(keywords[j])
        for item in list(level):
            keywords.remove(item)
        now_index += seq_cnt
        return keywords

    test_data['entity'] = test_data['entityExamples'].progress_apply(getLabel)
    test_data[['newsId', 'entity']].to_json("/mnt/souhu/code/ly/data_dir/entity_output.json",
                                            orient='records', lines=True, force_ascii=False)

'''
处理emotion输出
'''
def getResultForEmotion():
    res = []
    with open('data_dir/emotion_model/test_results.tsv', 'r') as f:
        for line in f.read().strip().split('\n'):
            example_out = line.strip()
            res.append(example_out)
    print("results文件读取完成")
    tqdm.pandas(desc="getLabel:")
    test_data = pd.read_json('data_dir/test_emotion.json', orient='records', lines=True)
    print("json文件读取完成")

    now_index = 0
    def getEmotion(example):
        emo_map = {'0': 'POS', '1': 'NORM', '2': 'NEG'}
        nonlocal now_index
        emotions = []
        for i in range(len(example)):
            emo = res[now_index]
            emotions.append(emo_map[emo])
            now_index += 1
        return emotions
    tqdm.pandas(desc="getLabel:")
    test_data['emotion'] = test_data['emotionExamples'].progress_apply(getEmotion)

    def getEntity(example):
        entities = []
        for i in range(len(example)):
            entities.append(example[i][0])
        return entities
    test_data['entity'] = test_data['emotionExamples'].progress_apply(getEntity)
    test_data[['newsId', 'entity',  'emotion']].to_json("/mnt/souhu/code/ly/data_dir/emotion_output.json",
                                                        orient='records', lines=True, force_ascii=False)

if __name__ == '__main__':
    getResultForEmotion()

