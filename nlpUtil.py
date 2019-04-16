#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-4-2 上午9:19
# @Author  : liyang
# @File    : divideWords.py

import pkuseg
import jieba
import jieba.posseg as pseg
from pyhanlp import *
from config import *
import os
from sgpyHandle import SgpyDictionary

class NLPUtil(object):
    _stopwords = [u'\n', u'，', u'。', u'、', u'：', u'(',
                  u')', u'[', u']', u'.', u',', u' ', u'\u3000', u'”', u'“',
                  u'？', u'?', u'！', u'‘', u'’', u'…', '!', '+', '-', '*',
                  '#', '"', "'", '^', ':', '/', '%', '=', ';', u'；', '@',
                  '{', '}', u'」', u'「', u'．', u'—', u'－', u'『', u'』',
                  u'□', u'【', u'】', u'◆', u'（', u'）', u'·', u'`', u'·',
                  u'\t', u'\\n', u'|']
    stopwords = set(_stopwords)
    jieba.load_userdict(SGPY_DICTIONARY)
    jieba.load_userdict(CHINESE_NAME_CORPUS)
    #seg = pkuseg.pkuseg(user_dict=SGPY_DICTIONARY, postag=True)
    seg = pkuseg.pkuseg(postag=True)
    #with open(SGPY_DICTIONARY) as f:
    #    for word in f.read().strip().split('\n'):
    #        CustomDictionary.add(word)

    @classmethod
    def divideWordWithPkuseg(cls, text):
        cuted_text = cls.seg.cut(text)
        res = ""
        for item in cuted_text:
            if item[0] not in cls.stopwords:
                res += item[0] + ':' + item[1] + ' '
        return res

    @classmethod
    def divideWordWithJieba(cls, text):
        cuted_text = pseg.lcut(text, HMM=True)
        res = ""
        for item, flag in cuted_text:
            if item not in cls.stopwords:
                res += item + ':' + flag + ' '
        return res

    @classmethod
    def divideWordWithHanLP(cls, text):
        cuted_text = HanLP.segment(text)
        res = ""
        for item in cuted_text:
            if item.word not in cls.stopwords:
                res += item.word + ':' + item.nature.name + ' '
        return res

if __name__=="__main__":
    text = '与牛共舞大数据研究中心建议关注：中海达、振芯科技、合众思壮、海格通信、北斗星通等。'
    res = NLPUtil.divideWordWithHanLP(text)
    print(res)