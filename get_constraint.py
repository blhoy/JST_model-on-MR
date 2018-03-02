#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs as cs
import re
from nltk.stem.porter import PorterStemmer

class DOC():

    def __init__(self):
        self.a = 0

    def read_text_f2(self, fname_list, samp_tag='text'):
        '''text format 2: one class one file, docs are sperated by samp_tag
        '''
        doc_str_list = []
        self.vocab = set()
        self.word_count = {}
        for fname in fname_list:
            doc_str = open(fname, 'r').read()
            patn = '<' + samp_tag + '>(.*?)</' + samp_tag + '>'
            str_list_one_class = re.findall(patn, doc_str, re.S)
            doc_str_list.extend(str_list_one_class)
        self.docs = []
        for x in doc_str_list:
            tt = []
            for item in x.split():
                item = porter.stem(item)
                if item in self.word_count:
                    self.word_count[item] += 1
                else:
                    self.word_count[item] = 0
                self.vocab.add(item)
                tt.append(item)
            self.docs.append(tt)
        return self.docs

    def read_sub(self, sub_path):
        file_str = open(sub_path, 'r').read()
        pre = 'word1='
        mid = ' pos1='
        patn = pre + '(.*?)' + mid
        word_list = re.findall(patn, file_str, re.S)

        bee = 'priorpolarity='
        patn = bee + '(.*?)' + '\n'
        polor = re.findall(patn, file_str, re.S)
        
        self.word_dict = {}
        self.word_dict2 = {}
        for index, word in enumerate(word_list):
            word = porter.stem(word)
            if polor[index] == 'negative':
                if word not in self.word_dict:
                    self.word_dict[word] = 0
                    self.word_dict2[word] = 0
                elif self.word_dict[word] != 0:
                    self.word_dict[word] = -2
                    self.word_dict2[word] = -2
            elif polor[index] == 'positive':
                if word not in self.word_dict:
                    self.word_dict[word] = 1
                    self.word_dict2[word] = 1
                elif self.word_dict[word] != 1:
                    self.word_dict[word] = -2
                    self.word_dict2[word] = -2
            else:
                if word not in self.word_dict:
                    self.word_dict[word] = -1
                    self.word_dict2[word] = -1
                elif self.word_dict[word] != -1:
                    self.word_dict[word] = -2
                    self.word_dict2[word] = -2

        for word in self.word_dict2:
            if word not in self.vocab or self.word_dict2[word] == -2:
                self.word_dict.pop(word)

        del self.word_dict2

    def write_full_constraint(self, path):
        f = cs.open(path, 'w')
        pos_num = 0
        neg_num = 0
        for word in self.word_dict:
            line = porter.stem(word)
            if self.word_dict[word] == 1:
                pos_num += 1
                line += '\t' + '0.05' + ' ' + '0.9' + ' 0.05' + '\n'
            elif self.word_dict[word] == 0:
                neg_num += 1
                line += '\t' + '0.05' + ' ' + '0.05' + ' 0.9' + '\n'
            else:
                continue
            f.write(line)
        f.close()
        print "pos word num : ",pos_num
        print "neg word num : ",neg_num

    def write_filter_constraint(self, path):
        f = cs.open(path, 'w')
        pos_num = 0
        neg_num = 0
        for word in self.word_dict:
            if self.word_count[word] >= 50:
                line = porter.stem(word)
                if self.word_dict[word] == 1:
                    pos_num += 1
                    line += '\t' + '0.05' + ' ' + '0.9' + ' 0.05' + '\n'
                elif self.word_dict[word] == 0:
                    neg_num += 1
                    line += '\t' + '0.05' + ' ' + '0.05' + ' 0.9' + '\n'
                else:
                    continue
                f.write(line)
        f.close()
        print "pos word num : ",pos_num
        print "neg word num : ",neg_num

porter = PorterStemmer()
filter_path = r'.\constraint\filter_lexicon.constraint'
path = r'.\constraint\full_subjectivity_lexicon.constraint'
sub_path = r'.\constraint\subjclueslen.tff'
corpus_path = r'.\data\MR'
fname_list = [r'.\data\neg',r'.\data\pos']
doc = DOC()
doc.read_text_f2(fname_list)
doc.read_sub(sub_path)
print len(doc.word_dict)
# doc.write_full_constraint(path)
doc.write_filter_constraint(filter_path)


#paradigm_words.constraint
# path = r'.\paradigm_words.constraint'

# f = cs.open(path, 'r')
# data = []
# for line in f.readlines():
#     head = line.split()[0]
#     porter = PorterStemmer()
#     head = porter.stem(head)
#     res = head + '\t' + ' '.join(line.split()[1:]) + '\n'
#     data.append(res)
# f.close()

# f = cs.open(path, 'w')
# for st in data:
#     f.write(st)
# f.close()