# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:18:42 2019

@author: Colouree
"""
import gensim
from gensim.models import Word2Vec

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
##    LOAD AND SAVE THE MODEL (GOOGLE NEWS VECTOR)      ###########################
#model =gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 
#model.save(r"C:\Users\Colouree\Desktop\Colouree\google_word2vec.model")
#####################################################################################
from gensim.models import KeyedVectors
model=KeyedVectors.load(r"C:\Users\Colouree\Desktop\Colouree\google_word2vec.model")

print(model.similarity('woman', 'man'))
print(model.n_similarity('queen woman', 'man'))