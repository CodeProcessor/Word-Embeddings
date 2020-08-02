'''
Created on 7/25/20

@author: dulanj
'''
import re
import string

from gensim.models import Word2Vec
from gensim.similarities.index import AnnoyIndexer


class Gensim(object):
    def __init__(self):
        self.lines = []
        self.text = None
        self.word_list = None

    def load_data(self):
        with open('sinhala.txt', 'r') as fp:
            self.text = fp.read()

    def clean_data(self):
        filtered_text = ''.join(
            [i for i in self.text if (not i.isdigit() and i not in set(string.punctuation) and i not in ['“', '”'])])
        re.sub(' +', ' ', filtered_text)
        sentences = filtered_text.split('\n')
        self.word_list = [line.split(' ') for line in sentences]
        print(self.word_list)

    def train_word_to_vec_v2(self, skipgram=0):
        # train model
        model = Word2Vec(self.word_list, size=100, window=5, min_count=3, workers=4, sg=skipgram)
        words = list(model.wv.vocab)
        if skipgram == 0:
            model_name = 'skipgram_model.bin'
        else:
            model_name = 'cbow_model.bin'
        model.save(model_name)
        # load model
        new_model = Word2Vec.load(model_name)
        # print(new_model)
        return new_model

    def similarities_check(self, model):


        # 100 trees are being used in this example
        annoy_index = AnnoyIndexer(model, 100)
        # Derive the vector for the word "science" in our model
        vector = model.wv["කටයුතු"]
        print(model.wv.most_similar([vector]))
        approximate_neighbors = model.wv.most_similar([vector], topn=11, indexer=annoy_index)
        print("Approximate Neighbors")
        for neighbor in approximate_neighbors:
            print(neighbor)

    def main(self):
        model_cbow = self.train_word_to_vec_v2(1)
        model_skipgram = self.train_word_to_vec_v2(0)
        self.similarities_check(model_cbow)


if __name__ == "__main__":
    obj = Gensim()
    obj.load_data()
    obj.clean_data()
    obj.main()
