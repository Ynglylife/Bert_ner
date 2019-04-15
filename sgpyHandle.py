#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-4-2 上午10:44
# @Author  : liyang
# @File    : sgpyHandle.py

#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import struct
import sys
import re
import time
from multiprocessing import Pool
import binascii
import pdb
from config import *

word_dir = '/media/liyang/Volume2/LY/sougouScel/'

# 搜狗的scel词库就是保存的文本的unicode编码，每两个字节一个字符（中文汉字或者英文字母）
# 找出其每部分的偏移位置即可
# 主要两部分
# 1.全局拼音表，貌似是所有的拼音组合，字典序
#       格式为(index,len,pinyin)的列表
#       index: 两个字节的整数 代表这个拼音的索引
#       len: 两个字节的整数 拼音的字节长度
#       pinyin: 当前的拼音，每个字符两个字节，总长len
#
# 2.汉语词组表
#       格式为(same,py_table_len,py_table,{word_len,word,ext_len,ext})的一个列表
#       same: 两个字节 整数 同音词数量
#       py_table_len:  两个字节 整数
#       py_table: 整数列表，每个整数两个字节,每个整数代表一个拼音的索引
#
#       word_len:两个字节 整数 代表中文词组字节数长度
#       word: 中文词组,每个中文汉字两个字节，总长度word_len
#       ext_len: 两个字节 整数 代表扩展信息的长度，好像都是10
#       ext: 扩展信息 前两个字节是一个整数(不知道是不是词频) 后八个字节全是0
#
#      {word_len,word,ext_len,ext} 一共重复same次 同音词 相同拼音表
class SgpyHandle(object):
    def __init__(self, filename):
        # 拼音表偏移，
        self.startPy = 0x1540
        # 汉语词组表偏移
        self.startChinese = 0x2628
        # 全局拼音表
        self.GPy_Table = {}
        # 解析结果
        # 元组(词频,拼音,中文词组)的列表
        self.GTable = []
        self.filen = filename

    def byte2str(self, data):
        '''将原始字节码转为字符串'''
        i = 0
        length = len(data)
        ret = u''
        while i < length:
            x = data[i:i+2]
            t = chr(struct.unpack('H', x)[0])
            if t == u'\r':
                ret += u'\n'
            elif t != u' ':
                ret += t
            i += 2
        return ret

    # 获取拼音表
    def getPyTable(self, data):
        if data[0:4] != bytes(map(ord,"\x9D\x01\x00\x00")):
            return None
        data = data[4:]
        pos = 0
        length = len(data)
        while pos < length:
            index = struct.unpack('H', data[pos:pos +2])[0]
            # print index,
            pos += 2
            l = struct.unpack('H', data[pos:pos + 2])[0]
            # print l,
            pos += 2
            py = self.byte2str(data[pos:pos + l])
            # print py
            self.GPy_Table[index] = py
            pos += l

    # 获取一个词组的拼音
    def getWordPy(self, data):
        pos = 0
        length = len(data)
        ret = u''
        while pos < length:
            index = struct.unpack('H', data[pos:pos + 2])[0]
            ret += self.GPy_Table[index]
            pos += 2
        return ret

    # 获取一个词组
    def getWord(self, data):
        pos = 0
        length = len(data)
        ret = u''
        while pos < length:
            index = struct.unpack('H', data[pos:pos +2])[0]
            ret += self.GPy_Table[index]
            pos += 2
        return ret

    # 读取中文表
    def getChinese(self, data):
        # import pdb
        # pdb.set_trace()

        pos = 0
        length = len(data)
        while pos < length:
            # 同音词数量
            same = struct.unpack('H', data[pos:pos + 2])[0]
            # print '[same]:',same,

            # 拼音索引表长度
            pos += 2
            py_table_len = struct.unpack('H', data[pos:pos + 2])[0]
            # 拼音索引表
            pos += 2
            py = self.getWordPy(data[pos: pos + py_table_len])

            # 中文词组
            pos += py_table_len
            for i in range(same):
                # 中文词组长度
                c_len = struct.unpack('H', data[pos:pos +2])[0]
                # 中文词组
                pos += 2
                word = self.byte2str(data[pos: pos + c_len])
                # 扩展数据长度
                pos += c_len
                ext_len = struct.unpack('H', data[pos:pos +2])[0]
                # 词频
                pos += 2
                count = struct.unpack('H', data[pos:pos +2])[0]

                # 保存
                self.GTable.append((count, py, word))

                # 到下个词的偏移位置
                pos += ext_len


    def deal(self):
        print('-' * 60)
        f = open(self.filen, 'rb')
        data = f.read()
        f.close()

        if data[0:12] != bytes(map(ord, "\x40\x15\x00\x00\x44\x43\x53\x01\x01\x00\x00\x00")):
            print("确认你选择的是搜狗(.scel)词库?")
            sys.exit(0)
        # pdb.set_trace()

        print("词库名：", self.byte2str(data[0x130:0x338]))  # .encode('GB18030')
        print("词库类型：", self.byte2str(data[0x338:0x540]))  # .encode('GB18030')
        print("描述信息：", self.byte2str(data[0x540:0xd40]))  # .encode('GB18030')
        print("词库示例：", self.byte2str(data[0xd40:self.startPy]))  # .encode('GB18030')

        self.getPyTable(data[self.startPy:self.startChinese])
        self.getChinese(data[self.startChinese:])
        self.print2Txt()

    def print2Txt(self):
        new_dir_pattern = re.compile(r'(.*/)')
        new_dir = new_dir_pattern.findall(self.filen)[0]
        pattern = re.compile(r'/sougouScel/')
        new_dir = pattern.sub('/sougouTxT/', new_dir)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        filename = self.filen.split('/')[-1]
        filename = filename.replace('scel', 'txt')
        new_file = new_dir + filename
        f = open(new_file, 'w', encoding='utf-8')
        for count, py, word in self.GTable:
            # GTable保存着结果，是一个列表，每个元素是一个元组(词频,拼音,中文词组)，有需要的话可以保存成自己需要个格式
            # 我没排序，所以结果是按照上面输入文件的顺序
            f.write('{%(count)s}' % {'count': count} + ' ' + py + ' ' + word)  # 最终保存文件的编码，可以自给改
            f.write('\n')
        f.close()

class SgpyDictionary(object):
    def __init__(self, category=None, limit=0):
        self.categories = category                                      # 默认为空，若输入，类型为list
        self.limit = limit                                              # 出现次数限制
        self.words_dir = '/media/liyang/Volume2/LY/sougouTxT/'                            # 词库目录

    def createDictionary(self,):
        start = time.time()
        dictionary = set()
        if self.categories == None:
            categories = os.listdir(self.words_dir)
        else:
            categories = self.categories
        for category in categories:
            cate_path = self.words_dir + category + '/'
            res = self.createDictionaryWithCategory(cate_path)
            dictionary = dictionary | res
        end = time.time()
        print('runs %0.2f seconds.' % (end - start))
        with open(SGPY_DICTIONARY, 'w', encoding='utf-8') as f:
            for word in list(dictionary):
                f.write(word + '\n')

    def createDictionaryWithMutiProc(self):
        start = time.time()
        dictionary = set()
        if self.categories == None:
            categories = os.listdir(self.words_dir)
        else:
            categories = self.categories
        procs = Pool(processes=8)
        categories = [self.words_dir + i + '/' for i in categories]
        res = procs.map(self.createDictionaryWithCategory, categories)
        for item in res:
            dictionary = dictionary | item
        procs.close()
        procs.join()
        end = time.time()
        print('runs %0.2f seconds.' % (end - start))
        print(len(list(dictionary)))
        with open(SGPY_DICTIONARY, 'w', encoding='utf-8') as f:
            for word in list(dictionary):
                f.write(word + '\n')

    def createDictionaryWithCategory(self, cate_path):
        dictionary = set()
        if os.path.isdir(cate_path):
            files = os.listdir(cate_path)
            for file in files:
                file_path = cate_path + file
                with open(file_path, 'r') as f:
                    for line in f.read().strip().split('\n'):
                        count, pinyin, word = line.split(' ')
                        count = int(count.replace('{', '').replace('}', ''))
                        if int(count) > self.limit:
                            dictionary.add(word)
        else:
            print("%s is not a directory!")
            exit(0)
        return dictionary

if __name__ == '__main__':
    dic = SgpyDictionary()
    dic.createDictionaryWithMutiProc()
    '''
    # 将要转换的词库添加在这里就可以了
    dirs = os.listdir(word_dir)
    for dir in dirs:
        dir_path = word_dir + dir
        if os.path.isdir(dir_path):
            dir_path += '/'
            files = os.listdir(dir_path)
            for file in files:
                file_path = dir_path + file
                handle = SgpyHandle(file_path)
                try:
                    handle.deal()
                except:
                    with open('exception.txt', 'a') as f:
                        f.write(file_path+'\n')
    '''


