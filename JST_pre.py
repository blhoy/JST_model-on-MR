#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-16 16:25:37
# @Author  : blhoy

""" Joint Sentiment-Topic Model using collapsed Gibbs sampling. """

import codecs as cs
import sys
import numpy as np
import numbers
import heapq

import _JST

class JST(object):

    def __init__(self, topics=1, sentilab=3, iteration=100, K=50,
             beta=0.01, gamma=0.01, random_state=123456789,
             refresh=50):
        self.topics = topics
        self.sentilab = sentilab
        self.iter = iteration
        self.alpha = (K + .0) / (topics + .0)
        self.beta = beta
        self.gamma = gamma
        self.random_state = random_state
        self.refresh = refresh

        if self.alpha <= 0 or beta <= 0 or gamma <= 0:
            raise ValueError("alpha,beta and gamma must be greater than zero")

        rng = self.check_random_state(random_state)
        self._rands = rng.rand(1024**2 // 8)

    def check_random_state(self, seed):
        if seed is None:
            # i.e., use existing RandomState
            return np.random.mtrand._rand
        if isinstance(seed, (numbers.Integral, np.integer)):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError("{} cannot be used as a random seed.".format(seed))

    def read_corpus(self, corpus_path):
        """Joint Sentiment-Topic Model using collapsed Gibbs sampling.

        In the corpus, every document takes one line, the first word of a document
        is 'di',where i is the index of the document,following are the words in the
        document sperated by ' '.

        """
        self.vocab = set()
        self.docs = []
        self.doc_num = 0
        self.doc_size = 0
        corpus = cs.open(corpus_path, 'r')
        for doc in corpus.readlines():
            self.doc_num += 1
            doc = doc.strip().split()[1:]
            for item in doc:
                self.vocab.add(item)
                self.doc_size += 1
            self.docs.append(doc)

        return self.docs

    def read_model_prior(self, prior_path):
        """Joint Sentiment-Topic Model using collapsed Gibbs sampling.

        Sentiment lexicon or other methods.

        The format of the prior imformation are as follow :
        [word]  [neu prior prob.] [pos prior prob.] [neg prior prob.]
        ...

        """
        self.prior = {}
        model_prior = cs.open(prior_path, 'r')
        for word_prior in model_prior.readlines():
            word_prior = word_prior.strip().split()
            index = 1
            maxm = -1.0
            for i in xrange(1,len(word_prior)):
                word_prior[i] = float(word_prior[i])
                if word_prior[i] > maxm:
                    maxm = word_prior[i]
                    index = i
            self.prior[word_prior[0]] = word_prior[1:]
            self.prior[word_prior[0]].append(index-1)

        return self.prior

    def analyzecorpus(self):
        #get dict {word:id ...} and {id:dict ...}

        self.vocabsize = len(self.vocab)
        if "d1950" in self.vocab:
            print "you are wrong !!!"
        print "vocabsize : ",self.vocabsize
        print "total doc words : ",self.doc_size
        self.word2id = {}
        self.id2word = {}
        index = 0
        for item in self.vocab:
            self.word2id[item] = index
            self.id2word[index] = item
            index += 1

    def init_model_parameters(self):
        #model counts
        self.nd = np.zeros((self.doc_num, ), dtype=np.int32)
        self.ndl = np.zeros((self.doc_num, self.sentilab), dtype=np.int32)
        self.ndlz = np.zeros((self.doc_num, self.sentilab, self.topics), dtype=np.int32)
        self.nlzw = np.zeros((self.sentilab, self.topics, self.vocabsize), dtype=np.int32)
        self.nlz = np.zeros((self.sentilab, self.topics), dtype=np.int32)

        #model parameters
        self.pi_dl = np.zeros((self.doc_num, self.sentilab), dtype=np.float)
        self.theta_dlz = np.zeros((self.doc_num, self.sentilab, self.topics), dtype=np.float)
        self.phi_lzw = np.zeros((self.sentilab, self.topics, self.vocabsize), dtype=np.float)

        #init hyperparameters with prior imformation
        self.alpha_lz = np.full((self.sentilab, self.topics), fill_value=self.alpha)
        self.alphasum_l = np.full((self.sentilab, ), fill_value=self.alpha*self.topics)

        if(self.beta <= 0):
            self.beta = 0.01
        # self.beta_lzw = np.full((self.sentilab, self.topics, self.vocabsize), fill_value=self.beta)
        # self.betasum_lz = np.full((self.sentilab, self.topics), fill_value=self.beta*self.vocabsize)
        self.beta_lzw = np.full((self.sentilab, self.topics, self.vocabsize), fill_value=self.beta)
        self.betasum_lz = np.zeros((self.sentilab, self.topics), dtype=np.float)

        # #word prior
        self.add_lw = np.ones((self.sentilab, self.vocabsize), dtype=np.float)

        self.add_prior()
        for l in xrange(self.sentilab):
            for z in xrange(self.topics):
                for r in xrange(self.vocabsize):
                    self.beta_lzw[l][z][r] *= self.add_lw[l][r]
                    self.betasum_lz[l][z] += self.beta_lzw[l][z][r]

        if self.gamma <= 0:
            self.gamma = 1.0
        # self.gamma_dl = np.full((self.doc_num, self.sentilab), fill_value=self.gamma)
        # self.gammasum_d = np.full((self.doc_num, ), fill_value=self.gamma*self.sentilab)

        self.gamma_dl = np.full((self.doc_num, self.sentilab), fill_value=0.0)
        self.gammasum_d = np.full((self.doc_num, ), fill_value=.0)
        for d in xrange(self.doc_num):
            # self.gamma_dl[d][1] = 1.8
            self.gamma_dl[d][1] = self.gamma
            self.gamma_dl[d][2] = self.gamma
        for d in xrange(self.doc_num):
            for l in xrange(self.sentilab):
                self.gammasum_d[d] += self.gamma_dl[d][l]

    def add_prior(self):
        #beta add prior imformation
        for word in self.prior:
            if word in self.vocab:
                # label = self.prior[word][-1]
                for l in xrange(self.sentilab):
                    # if l == label:
                    self.add_lw[l][self.word2id[word]] *= self.prior[word][l]
                    #     self.add_lw[l][self.word2id[word]] = 1.0
                    # else:
                    #     self.add_lw[l][self.word2id[word]] = 0.0

    def init_estimate(self):
        print "Estimate initializing ..."
        self.ZS = []
        self.LS = []
        self.WS = []
        self.DS = []
        self.IS = []

        cnt = 1
        prior_word_cnt = 0
        for m, doc in enumerate(self.docs):
            for t, word in enumerate(doc):
                cnt += 1
                if word in self.prior:
                    senti = self.prior[word][-1]
                    self.IS.append(int(1))
                    prior_word_cnt += 1
                else:
                    # senti = int(np.random.uniform(0,self.sentilab))
                    senti = (cnt) % self.sentilab
                    self.IS.append(int(0))
                # topi = int(np.random.uniform(0,self.topics))
                topi = (cnt) % self.topics
                self.DS.append(int(m))
                self.WS.append(int(self.word2id[word]))
                self.LS.append(int(senti))
                self.ZS.append(int(topi))

                self.nd[m] += 1
                self.ndl[m][senti] += 1
                self.ndlz[m][senti][topi] += 1
                self.nlzw[senti][topi][self.word2id[word]] += 1
                self.nlz[senti][topi] += 1

        self.DS = np.array(self.DS, dtype = np.int32)
        self.WS = np.array(self.WS, dtype = np.int32)
        self.LS = np.array(self.LS, dtype = np.int32)
        self.ZS = np.array(self.ZS, dtype = np.int32)
        self.IS = np.array(self.IS, dtype = np.int32)
        print "DS number and cnt are : ", len(self.DS),' ',cnt - 1
        print "affected words : ",prior_word_cnt

        print "total word # is : ", cnt - 1

    def estimate(self):
        random_state = self.check_random_state(self.random_state)
        rands = self._rands.copy()
        # print rands[:100]
        self.init_estimate()
        print "set topics : ",self.topics
        print "set gamma : ",self.gamma_dl[0][1], self.gamma_dl[0][2]
        print "The {} iteration of sampling ...".format(self.iter)
        ll = ll_pre = 0.0
        for it in xrange(self.iter):
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                print "Iteration {} :".format(it)
                ll += self.loglikelihood()
                print "<{}> log likelihood: {:.0f}".format(it, ll/(it/self.refresh + 1))
                if ll/(it/self.refresh + 1) - 10 <= ll_pre and it > 0:
                    break
                ll_pre = ll/(it/self.refresh + 1)
            self._sampling(rands)

    def loglikelihood(self):
        """Calculate complete log likelihood, log p(w,z,l)

        Formula used is log p(w,z,l) = log p(w|z,l) + log p(z|l,d) + log p(l|d)
        """
        nd, ndl, ndlz, nlzw, nlz = self.nd, self.ndl, self.ndlz, self.nlzw, self.nlz
        return _JST._loglikelihood(nd, ndl, ndlz, nlzw, nlz, self.alpha, self.beta, self.gamma)

    def _sampling(self, rands):

        _JST._sample_topics(self.nd, self.ndl, self.ndlz, self.nlzw, self.nlz,
                self.alpha_lz, self.alphasum_l, self.beta_lzw, self.betasum_lz,
                self.gamma_dl, self.gammasum_d, self.DS, self.WS, self.LS, self.ZS, self.IS,
                rands)

    def cal_pi_ld(self):
        for d in xrange(self.doc_num):
            for l in xrange(self.sentilab):
                self.pi_dl[d][l] = (self.ndl[d][l] + self.gamma_dl[d][l]) / (self.nd[d] + self.gammasum_d[d])

    def cal_theta_dlz(self):
        for d in xrange(self.doc_num):
            for l in xrange(self.sentilab):
                for z in xrange(self.topics):
                    self.theta_dlz[d][l][z] = (self.ndlz[d][l][z] + self.alpha_lz[l][z]) \
                    / (self.ndl[d][l] + self.alphasum_l[l])

    def cal_phi_lzw(self):
        for l in xrange(self.sentilab):
            for z in xrange(self.topics):
                for w in xrange(self.vocabsize):
                    self.phi_lzw[l][z][w] = (self.nlzw[l][z][w] + self.beta_lzw[l][z][w]) \
                    / (self.nlz[l][z] + self.betasum_lz[l][z])

def main():
    test = JST()
    test.read_corpus(r'.\data\MR')
    
    if len(sys.argv) > 1 and sys.argv[1] == '1':
        test.read_model_prior(r'.\constraint\mpqa.constraint')
    elif len(sys.argv) > 1 and sys.argv[1] == '2':
        test.read_model_prior(r'.\constraint\paradigm_words.constraint')
    elif len(sys.argv) > 1 and sys.argv[1] == '3':
        test.read_model_prior(r'.\constraint\full_subjectivity_lexicon.constraint')
    elif len(sys.argv) > 1 and sys.argv[1] == '4':
        test.read_model_prior(r'.\constraint\filter_lexicon.constraint')
    elif len(sys.argv) > 1 and sys.argv[1] == '0':
        test.read_model_prior(r'.\constraint\empty_prior')
    else:
        print "number should be 0 ~ 4"
        return
    
    # print test.docs[1][:10]
    # print len(test.vocab),' ',test.prior['happi']
    # test.read_model_prior(r'.\constraint\filter_lexicon.constraint')
    test.analyzecorpus()
    test.init_model_parameters()
    test.estimate()

    test.cal_pi_ld()
    t1 = np.min(test.pi_dl[:,1])
    t2 = np.min(test.pi_dl[:,2])
    print "PI_min is : ", min(t1, t2)
    cnt_pos = cnt_neg = 0
    doc_pos = 0
    doc_neg = 0

    for d in xrange(test.doc_num):
        # if(d < 10):
        #     print test.pi_dl[d]
        if test.pi_dl[d][1] > test.pi_dl[d][2]:
            doc_pos += 1
        if(test.pi_dl[d][1] <= test.pi_dl[d][2]):
            doc_neg += 1
        if(d < 1000 and test.pi_dl[d][1] <= test.pi_dl[d][2] ):
            cnt_neg += 1
        elif(d >= 1000 and test.pi_dl[d][1] > test.pi_dl[d][2]):
            cnt_pos += 1
    print "doc_neg : ",doc_neg,' ',"doc_pos : ",doc_pos

    # for i in xrange(test.doc_num):
    #     if(i < 1000 and test.ndl[i][1] < test.ndl[i][2] ):
    #         cnt_neg += 1
    #     elif(i >= 1000 and test.ndl[i][1] > test.ndl[i][2]):
    #         cnt_pos += 1

    print "pos accurancy is : {:.2f}%".format((cnt_pos + .0) / test.doc_num * 200)
    print "neg accurancy is : {:.2f}%".format((cnt_neg + .0) / test.doc_num * 200)
    print "total accurancy is : {:.2f}%".format((cnt_pos + cnt_neg + .0) / test.doc_num * 100)

    # pp = (cnt_pos + .0) / doc_pos * 100
    # pn = (cnt_neg + .0) / doc_neg * 100
    # print "pos accurancy is : {:.2f}%".format(pp)
    # print "neg accurancy is : {:.2f}%".format(pn)
    # print "avrage accurancy is : {:.2f}%".format((pp + pn + .0) / 2)
    # print "total accurancy is : {:.2f}%".format((cnt_pos + cnt_neg + .0) / test.doc_num * 100)

    # print test.DS[:30]
    # print test.LS[:30]
    # print test.ZS[:30]
    test.cal_phi_lzw()

    for l in xrange(test.sentilab):
        print "sentiment ",l," :"
        for z in xrange(test.topics):
            res = heapq.nlargest(20, xrange(len(test.phi_lzw[l][z])), test.phi_lzw[l][z].take)
            print "\ntopic ",z," : ",
            for item in res:
                print test.id2word[item],' ',
        print "\n*************************************"


if __name__ == '__main__':
    main()