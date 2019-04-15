#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-4-2 上午10:04
# @Author  : liyang
# @File    : config.py

DATA_DIR = '../../data/'
TRAIN_DATA = DATA_DIR + 'coreEntityEmotion_train.txt'
TEST_DATA = DATA_DIR + 'coreEntityEmotion_test_stage1.txt'
ENTITY_WORDS = DATA_DIR + 'token.words'
TRAIN_EXTRACTS = DATA_DIR + 'coreEntityEmotionExtracts.csv'
TRAIN_EXTRACTS_SEG = DATA_DIR + 'coreEntityEmotionExtractsSeg_%s.csv'
SGPY_DICTIONARY = DATA_DIR + 'dictionary.txt'
CHINESE_NAME_CORPUS = DATA_DIR + 'Chinese_Names_Corpus（120W）.txt'
TRAIN_EXTRACTS_SEG_json = DATA_DIR + 'coreEntityEmotionExtractsSeg_%s.json'