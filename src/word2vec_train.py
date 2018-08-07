# -*- coding: utf-8 -*-
import os.path
import environment
import sqlite3
import utils
from nltk import sent_tokenize, RegexpTokenizer
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import tokenize

class SQLiteCorpus(object):
    def __init__(self, path_to_db, limit=None, random=False, bigrams=False):
        self.dbpath = path_to_db
        self.limit = limit
        self.random = random
        self.bigrams = self.computeBigrams() if bigrams else None
 
    def __iter__(self):
        for review in self.getReviews():
            review = self.preprocess(review[0])
            for sentence in self.getSentences(review):
                unigrams = self.getUnigrams(sentence)
                yield self.bigrams[unigrams] if self.bigrams else unigrams
    
    def getReviews(self):
        '''
        Get reviews from the db (optionally limiting and randomizing)
        '''
        with sqlite3.connect(self.dbpath) as conn:
            conn.text_factory = bytes
            c = conn.cursor()
            query = "SELECT content FROM content"
            query += " ORDER BY RANDOM()" if self.random else ""
            query += " LIMIT {}".format(self.limit) if self.limit else ""
            reviews_list = c.execute(query)
        return reviews_list

    def getSentences(self, review):
        # return utils.splitIntoSentences(review)
        return sent_tokenize(unicode(review, "utf-8"))

    def getUnigrams(self, sentence):
        # return list(tokenize(sentence))
        tokenizer = RegexpTokenizer(r'\w+')
        return tokenizer.tokenize(sentence)

    def computeBigrams(self):
        reviews = self.getReviews()
        sentences = []
        for review in reviews:
            review = self.preprocess(review[0])
            sentences += self.getSentences(review)
        stream = [self.getUnigrams(sentence) for sentence in sentences]
        phrases = Phrases(stream, min_count=10, scoring="npmi")
        return Phraser(phrases)

    def preprocess(self, text):
        # replace fancy characters
        text = utils.replaceChars(text)
        text = utils.expandContractions(text=text.lower())
        return text

if __name__ == "__main__":
    path = environment.PATH
    databasedir = environment.DB_DIR
    databasepath = environment.DB_PATH
    modelsdir = environment.MODELS_DIR
    print(path)
    
    # Train simple W2V model
    #modelname = "word2vec_sg.model"
    #corpus = SQLiteCorpus(databasepath)  # a memory-friendly iterator
    #model = Word2Vec(corpus, sg=1, workers=4)
    #model.save(modelsdir + modelname)
    #if not os.path.isfile(modelsdir + modelname):
    #    print("W2V model trained.")
    #else:
    #    print("W2V model already exists! Overwriting...")

    # Train W2V model using bigrams
    modelname = "word2vec_sg_bigrams.model"
    corpus = SQLiteCorpus(databasepath, bigrams=True)
    model = Word2Vec(corpus, sg=1, workers=4, size=200)
    model.wv.save_word2vec_format(modelsdir+modelname, binary=True)
    #model.save(modelsdir + modelname)
    
    if not os.path.isfile(modelsdir + modelname):
        print("W2V bigrams model trained.")
    else:
        print("W2V bigrams model already exists!")
