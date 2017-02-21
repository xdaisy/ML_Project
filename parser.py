import pandas as pd
import numpy as np

class Parser(object):
    def __init__(self):
        self.df = pd.DataFrame()

    def parse(self):
        tfidf_file = open("tfidf_test.txt")
        tfidf = tfidf_file.read()
        print tfidf


    def parse_article(self, article):
        pass

if __name__ == "__main__":
    p = Parser()
    p.parse()