#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, sys, string
import codecs as cs
from nltk.stem.porter import PorterStemmer

def read_text_f2(fname_list, samp_tag='text'):
    '''text format 2: one class one file, docs are sperated by samp_tag
    '''
    doc_str_list = []
    for fname in fname_list:
        doc_str = open(fname, 'r').read()
        patn = '<' + samp_tag + '>(.*?)</' + samp_tag + '>'
        str_list_one_class = re.findall(patn, doc_str, re.S)
        doc_str_list.extend(str_list_one_class)
    delEStr = string.punctuation + string.digits
    docs = []
    for x in doc_str_list:
        for c in delEStr:
            x = x.replace(c, '')
            tt = [item for item in x.split() if item.isalpha()]
        
        docs.append(tt)
    return docs

def remove_stop_words(stopwords_file, ori_docs):
    stopwords = [x.strip() for x in open(stopwords_file).readlines()]
    docs = []
    for doc in ori_docs:
        arti = []
        for word in doc:
            if word not in stopwords:
                arti.append(word)
        docs.append(arti)
    return docs

def word_porter(doc_unis_list):
    doc_stems_list = []
    porter = PorterStemmer()
    cnt = 0
    for doc_unis in doc_unis_list:
        # print len(doc_unis),' ',cnt
        doc_stems = []
        for word in doc_unis:
            cnt += 1
            stem_uni = porter.stem(word)
            doc_stems.append(stem_uni)
        doc_stems_list.append(doc_stems)
    print "now total words: ",cnt
    return doc_stems_list

if __name__ == '__main__':
    fname_list = [r".\data\neg", r".\data\pos"]
    stopwords = r".\constraint\stop_words.txt"
    doc_str_list = read_text_f2(fname_list)
    print len(doc_str_list)
    ori_docs = [x for x in doc_str_list]
    docs = remove_stop_words(stopwords, ori_docs)
    
    print "ok ..."
    # print len(docs),"\n",docs[0]
    doc_final = word_porter(docs)
    print "preprocessing done ..."

    word_count = 0
    terms = set()
    f = cs.open(r'.\data\MR', 'w')
    porter = PorterStemmer()
    for i, doc in enumerate(doc_final):
        line_str = 'd' + str(i) + '\t' + ' '.join([porter.stem(word) for word in doc]) + '\n'
        f.write(line_str)
        for word in doc:
            word_count += 1
            terms.add(word)
    print "word_count : ",word_count
    print "distinct terms : ",len(terms)
    