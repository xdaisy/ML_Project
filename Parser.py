import pandas as pd
import numpy as np
import mysql

DICT_LEN = 10575

class Parser(object):
    def __init__(self, file_name):
        self.df = None
        self.file_name = file_name

    def parse(self):
        """parses the article in self.file_name
        Returns a pandas dataframe"""
        tfidf_file = open(self.file_name)
        tfidf = tfidf_file.read().strip().split("\n")
        labels = []
        for i in range(DICT_LEN):
            labels.append(i)
        rows = []
        for art in tfidf:
            rows.append(art.split("|")[0])
        self.df = pd.DataFrame(index=rows, columns=pd.Series(labels))
        # print labels
        # print pd.Series(labels)
        i = 0
        for article in tfidf:
            article_name, data = self.parse_article(article)
            self.df.loc[article_name] = data
            print i
            i += 1
        return self.df.astype(float)


    def parse_article(self, article):
        """parses a single row of self.file_name
        Returns an array of tfidf values"""
        row = [0] * (DICT_LEN)
        article_comps = article.split("|")
        data = article_comps[1].split(",")
        for datum in data:
            word, val = datum.split(":")
            row[int(word)] = float(val)
        return article_comps[0], row

if __name__ == "__main__":
    p = Parser("tfidf_test.txt")
    print(p.parse().dtypes)