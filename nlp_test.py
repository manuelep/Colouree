#class NLP:
#    def __init__(self):
        





#from fuzzywuzzy import fuzz
#k=fuzz.token_set_ratio('Deluxe Room, 1 King Bed', 'Hotel')
##import nltk 
##nltk.download('wordnet')
##
##from nltk.corpus import wordnet
##
##list1 = ['Compare', 'require']
##list2 = ['choose', 'copy', 'define', 'duplicate', 'find', 'how', 'identify', 'label', 'list', 'listen', 'locate', 'match', 'memorise', 'name', 'observe', 'omit', 'quote', 'read', 'recall', 'recite', 'recognise', 'record', 'relate', 'remember', 'repeat', 'reproduce', 'retell', 'select', 'show', 'spell', 'state', 'tell', 'trace', 'write']
##
##from gensim.test.utils import common_texts, get_tmpfile
##from gensim.models import Word2Vec
##
##path = get_tmpfile("word2vec.model")
##
##model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
##model.save("word2vec.model")
##
##
##
##
##
#
#
#
#
#
#
#from gensim.models import Word2Vec
##model = Word2Vec.load("word2vec.model")
##model.similarity('france', 'spain')
#import gensim
#import gzip
#data_file="reviews_data.txt.gz"
#
##with gzip.open ('reviews_data.txt.gz', 'rb') as f:
##    for i,line in enumerate (f):
##        print(line)
##        break
#def read_input(input_file):
#    """This method reads the input file which is in gzip format"""
#    
##    logging.info("reading file {0}...this may take a while".format(input_file))
#    
#    with gzip.open (input_file, 'rb') as f:
#        for i, line in enumerate (f): 
#
##            if (i%10000==0):
#                
##                logging.info ("read {0} reviews".format (i))
#            # do some pre-processing and return a list of words for each review text
#            yield gensim.utils.simple_preprocess (line)
#
## read the tokenized reviews into a list
## each review item becomes a serries of words
## so this becomes a list of lists
#documents = read_input (data_file)
#
##model = gensim.models.Word2Vec (documents, size=150, window=10, min_count=2, workers=10)
##model.train(documents,total_examples=len(documents),epochs=10)
#
#
##model.wv.similarity(w1="dirty",w2="smelly")
#
#import fastsemsim
#str1='Birthday party ruined as cake explodes'
#str2='Grandma mistakenly bakes cake using gunpowder'
#k=fastsemsim.semsim..similarity(str1,str2)
#
#
#
#
#from gensim import corpora, models, similarities

#!/usr/bin/python

#import codecs
#import os
#import shutil
#
#SOURCE = 'glove.840B.300d.txt'
#TARGET = 'glove.840B.300d--modified.txt'
#TMP = 'tmp.file'
#
#LENGTH_BY_PREFIX = [
#  (0xC0, 2), # first byte mask, total codepoint length
#  (0xE0, 3), 
#  (0xF0, 4),
#  (0xF8, 5),
#  (0xFC, 6),
#]
#
#def codepoint_length(first_byte):
#    if first_byte < 128:
#        return 1 # ASCII
#    for mask, length in LENGTH_BY_PREFIX:
#        if first_byte & mask == mask:
#            return length
#        else:
#            return 0
#
#def read_utf8_char_and_decode(source):
#    c = source.read(1)
#    if c:
#        char_len = codepoint_length(ord(c))
#    else:
#        return u''
#    if char_len:
#        c += source.read(char_len-1)
#        try:
#            c=c.decode('utf8')
#        except:
#            return u''
#    else:
#        return u''
#    return c
#
#source = open(SOURCE,mode='r')
#print(source)
#tmp = codecs.open(TMP,mode='w',encoding='utf8')
#line = source.readline()
#vsize, nbdim = line.split()
#vsize = int(vsize)
##print vsize
#count = 0
#bad = 0
#i = 0
#wrong_chars = [u'',u'\u00A0',u'\u2026',u'\u000A', u'\u000B', u'\u000C', u'\u000D', u'\u0085', u'\u2028', u'\u2029']
##print "Started ..."
#while i<vsize:
#    if i % 100000 == 0:
#        print (i)
#    i+=1
#    s = u''
#    c = u''
#    while c != u' ':
#        c = read_utf8_char_and_decode(source)
#        if c in wrong_chars:
#            if c:
#                print('Error %s') %repr(c)
#            bad+=1
#            source.readline()
#            break
#        else:
#            s += c
#    if c in wrong_chars:
#        continue
#    s2 = source.readline()
#    try:
#        s2 = s2.decode('utf8')
#    except:
##        print "Error: %s" % s2
#        bad += 1
#        continue
#    count += 1
#    tmp.write(s+s2)
#
##print "%d bad words" % bad
##print "%d total word count" % count
##print "Now copying to the target file..."
#
#source.close()
#tmp.close()
#
#with codecs.open(TMP,mode='r',encoding='utf8') as tmp:
#    with codecs.open(TARGET,mode='w',encoding='utf8') as target:
#        target.write("%d 300\n" % (count))
#        shutil.copyfileobj(tmp, target)
#
#tmp.close()
#target.close()
#os.remove(TMP)
#
#print("Done.")

#```python
#from sematch.semantic.similarity import WordNetSimilarity
#wns = WordNetSimilarity()
#
## Computing English word similarity using Li method
#wns.word_similarity('dog', 'cat') # 0.449327301063
##
#def filter_words(words):
#    if words is None:
#        return
#    return [word for word in words if word in model.vocab]

import gensim
import pandas as pd
df=pd.read_csv('only_amenity_tags.csv',encoding='latin-1')
amenity=df['amenity']
#amenity=df.loc[df['amenity'] == 'social_facility']['value']
import re

regex = re.compile('[^a-zA-Z]')

#l = ["a#bc1!","a(b$c"]

keys=[regex.sub(' ', i) for i in amenity]

def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)+' '
    return result

#print(concatenate_list_data([1, 5, 12, 2]))
    
keys=list(set(keys))
##lll=[x for x in keys if 'wheelchair' in x]
keys=[concatenate_list_data(k.split(' ')[0:3]) for k in keys]
keys=[k.replace('centre','center').replace('of','').replace('theatre','theater') for k in keys]

macro_tags=['education',
'health',
'residential',
'culture', 'culture leisure',
'hospitality', 'tourism infrastructure','tourism','infrastructure',
'food drink',
'government',
'sport wellness',
'transportation',
'public spaces',
'services','business',
'business workplace',
'commercial shops',
'worship','living','personal service','personal','leisure','mobility service','work']
## Load Google's pre-trained Word2Vec model.
#model_gn = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
#
##keys = ['Paris', 'Python', 'Sunday', 'Tolstoy', 'Twitter', 'bachelor', 'delivery', 'election', 'expensive',
##        'experience', 'financial', 'food', 'iOS', 'peace', 'release', 'war']
#keys=[x.replace('_',' ').replace(':',' ').replace(';',' ').replace('.',' ').replace('/','').replace('\\','') for x in amenity]# if not '_' in x]#.replace('_',' ')
#keys=df.keys.str.replace('[^a-zA-Z0-9]', '')
#keys=[x for x in keys if not 'theatre' in x]
#embedding_clusters = []
#word_clusters = []
#for word in keys:
#    embeddings = []
#    words = []
#    for similar_word, _ in model_gn.most_similar(word, topn=30):
#        words.append(similar_word)
#        embeddings.append(model_gn[similar_word])
#    embedding_clusters.append(embeddings)
#    word_clusters.append(words)
#from sklearn.manifold import TSNE
#import numpy as np
#
#embedding_clusters = np.array(embedding_clusters)
#n, m, k = embedding_clusters.shape
#tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
#embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
#
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
##% matplotlib inline
#
#
#def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
#    plt.figure(figsize=(100, 100))
#    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
#    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
#        x = embeddings[:, 0]
#        y = embeddings[:, 1]
#        plt.scatter(x, y, c=color, alpha=a, label=label)
#        for i, word in enumerate(words):
#            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
#                         textcoords='offset points', ha='right', va='bottom', size=8)
#    plt.legend(loc=4)
#    plt.title(title)
#    plt.grid(True)
#    if filename:
#        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
#    plt.show()
#
#
#tsne_plot_similar_words('Similar words from Google News', keys, embeddings_en_2d, word_clusters, 0.7,
#                        'similar_words.png')
#




#
#import gensim.downloader as api
#word_vectors=api.load("glove-wiki-gigaword-100")
#similarity=word_vectors.similarity('pharmacy','hospital')
#from gensim.models import Word2Vec


#model = Word2Vec.load(r"C:\Users\Colouree\Desktop\Colouree\word2vec.model")
import time
start = time.time()
####from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
####from gensim.scripts.glove2word2vec import glove2word2vec
####
####
#####glove_file = datapath(r'C:\Users\Colouree\Desktop\Colouree\glove.840B.300d.txt')
#####tmp_file = get_tmpfile(r"glove.840B.300d_word2vec.txt")
#####_ = glove2word2vec(glove_file, tmp_file)
#####model = KeyedVectors.load_word2vec_format(tmp_file)
#####model.save(r"C:\Users\Colouree\Desktop\Colouree\word2vec.model")
#model=KeyedVectors.load(r"C:\Users\Colouree\Desktop\Colouree\word2vec.model")
model1=KeyedVectors.load(r"C:\Users\Colouree\Desktop\Colouree\google_word2vec.model")
print("took {} secs to load the model".format(time.time()-start))
#start = time.time()
#def compare_two_words(tag1,tag2):
#    result=model.similarity(tag1, tag2)
#    return result

final_tags=[]
for word in keys:
    x=''
    for ij in word.split():
        if ij in model1.vocab:
            x+=ij+' '
    if not x=='':
        final_tags.append(x)
import pandas as pd
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))

df1=pd.DataFrame(columns=final_tags,index=macro_tags)
#from tqdm import tqdm
#for i in tqdm(range(0,len(macro_tags))):
#    for j in range(0,len(final_tags)):
#        try:
#
#            df1.iloc[i,j]=((model.similarity(df1.index[i], df1.columns[j])+model1.similarity(df1.index[i], df1.columns[j]))+0.0000001)/2
#        except:
#            try:
#                df1.iloc[i,j]=((model.n_similarity(df1.index[i].lower().split(), df1.columns[j].lower().split())+model1.n_similarity(df1.index[i].lower().split(), df1.columns[j].lower().split()))+0.00001)/2
##                first=df1.index[i].lower().split()
##                second=df1.index[j].lower().split()
##                df1.iloc[i,j]=((model.n_similarity(first, second)+model1.n_similarity(first, second))+0.00001)/2
##                try:
##                    first= [w for w in df1.index[i].lower().split() if not w in stop_words]
##                except:
##                    first=df1.index[i].lower().split()
##                try:
##                    second= [w for w in df1.index[j].lower().split() if not w in stop_words]
##                except:
##                    second=df1.index[j].lower().split()
##                try:
##                    df1.iloc[i,j]=((model.n_similarity(first, second)+model1.n_similarity(first, second))+0.00001)/2
##                except:
##                    df1.iloc[i,j]=((model.similarity(first, second)+model1.n_similarity(first, second))+0.00001)/2
#            except:
#                df1.iloc[i,j]=0.0
df02=pd.DataFrame(columns=final_tags,index=macro_tags)
from tqdm import tqdm
for i in tqdm(range(0,len(macro_tags))):
    for j in range(0,len(final_tags)):
        try:

            df02.iloc[i,j]=model1.similarity(df02.index[i], df02.columns[j])
        except:
            try:
                df02.iloc[i,j]=model1.n_similarity(df02.index[i].lower().split(), df02.columns[j].lower().split())
            except:
                df02.iloc[i,j]=0.0
                
                
#from tqdm import tqdm
#for i in tqdm(range(0,len(macro_tags))):
#    for j in range(0,len(final_tags)):
#        try:
#
#            df1.iloc[i,j]=model.similarity(df1.index[i], df1.columns[j])
#        except:
#            try:
#                df1.iloc[i,j]=model.n_similarity(df1.index[i].lower().split(), df1.columns[j].lower().split())
#            except:
#                df1.iloc[i,j]=0.0
##for i in keys:
##    df2['%s'.format(i)]=df1['%s'.format(i)]


def get_nearest_tags(tag,df1,nearest_words):
    df1=df1.astype('float32')
#    current_tag=df1.loc[tag].sort_values(ascending=False)[1:nearest_words+1]
#    current_tag=cr_df.loc[cr_df.index.values.astype(str)[0]].sort_values(ascending=False)[1:nearest_words+1]
#    df2=pd.DataFrame(current_tag,columns=tag,index=current_tag.index)
    return df1.loc[tag].sort_values(ascending=False)[1:nearest_words+1]

def unknown_tag(tag_name,macro_tags,*args):
    if tag_name not in final_tags:
        final_tags.append(tag_name) 
    df02=pd.DataFrame(columns=final_tags,index=macro_tags)
    from tqdm import tqdm
    for i in tqdm(range(0,len(macro_tags))):
        for j in range(0,len(final_tags)):
            try:
    
                df02.iloc[i,j]=model1.similarity(df02.index[i], df02.columns[j])
            except:
                try:
                    df02.iloc[i,j]=model1.n_similarity(df02.index[i].lower().split(), df02.columns[j].lower().split())
                except:
                    df02.iloc[i,j]=0.0
    return df02[tag_name]
def khudka_tag(tag_name,nearest_words):
    import pandas as pd
    df=pd.read_csv('only_amenity_tags.csv',encoding='latin-1')
    amenity=df['amenity'].unique()
    
    ## Load Google's pre-trained Word2Vec model.
    #model_gn = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
    ##keys = ['Paris', 'Python', 'Sunday', 'Tolstoy', 'Twitter', 'bachelor', 'delivery', 'election', 'expensive',
    ##        'experience', 'financial', 'food', 'iOS', 'peace', 'release', 'war']
    keys=[x.replace('_',' ').replace(':',' ').replace(';',' ') for x in amenity]
    keys.append(tag_name)
    df1=pd.DataFrame(columns=keys,index=keys)
    for i in range(0,len(keys)):
        for j in range(0,len(keys)):
            try:
                df1.iloc[i,j]=((model1.similarity(df1.index[i], df1.columns[j])+model1.similarity(df1.index[i], df1.columns[j]))+0.0000001)/2
            except:
                try:
                    df1.iloc[i,j]=((model1.n_similarity(df1.index[i].lower().split(), df1.columns[j].lower().split())+model1.n_similarity(df1.index[i].lower().split(), df1.columns[j].lower().split()))+0.00001)/2
                except:
                    df1.iloc[i,j]=0.0
    result=get_nearest_tags(tag_name,df1,nearest_words)
    return result

def andarka_tag(tag_name,keys,nearest_words):
    cr_df=pd.DataFrame(columns=keys,index=[tag_name])
#    import pandas as pd
#    df=pd.read_csv('all_the_tags.csv',encoding='latin-1')
#    amenity=df['key'].unique()
#    #
#    ## Load Google's pre-trained Word2Vec model.
#    #model_gn = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
#    ##keys = ['Paris', 'Python', 'Sunday', 'Tolstoy', 'Twitter', 'bachelor', 'delivery', 'election', 'expensive',
#    ##        'experience', 'financial', 'food', 'iOS', 'peace', 'release', 'war']
#    keys=[x.replace('_',' ').replace(':',' ').replace(';',' ') for x in amenity]
#    keys.append(tag_name)
#    df1=pd.DataFrame(columns=keys,index=keys)
#    for i in range(0,len(keys)):
    i=0
    for j in range(0,len(keys)):
        try:
            cr_df.iloc[i,j]=model1.similarity(cr_df.index[i], cr_df.columns[j])
        except:
            try:
                cr_df.iloc[i,j]=model1.n_similarity(cr_df.index[i].lower().split(), cr_df.columns[j].lower().split())
            except:
                cr_df.iloc[i,j]=0.0
    result=get_nearest_tags(tag_name,cr_df,nearest_words)
    return result,cr_df

print("took {} secs to run the analysis".format(time.time()-start))
df1.to_csv('newest_tags_relations1.csv')
df02.to_csv('newest_tags_relations2.csv')
#result,cr_df=andarka_tag('living',keys,200)
#education
#health
#residential
#culture & leisure
#hospitality, tourism infrastructure
#food & drink
#government
#sport & wellness
#transportation
#public spaces
#services
#business/workplace
#commercial/shops
#worship


#import matplotlib.pyplot as plt
#import numpy as np
#correlations = df1.astype('float32').corr()
## plot correlation matrix
#fig = plt.figure(figsize=(50, 50))
#ax = fig.add_subplot(111)
#cax = ax.matshow(correlations, vmin=-1, vmax=1)
#fig.colorbar(cax)
#ticksx = np.arange(0,len(df1.index),1)
#ticksy = np.arange(0,len(df1.columns),1)
#ax.set_xticks(ticksx)
#ax.set_yticks(ticksy)
#ax.set_xticklabels(df1.index,rotation=45)
#ax.set_yticklabels(df1.columns)
#plt.show()
#plt.savefig('correlation_matrix.png')



















#from gensim.models import Word2Vec
#kkk=model.n_similarity(filter_words('Farmacia Operaia Sormani'),filter_words('Farmacia Cappuccini'))
#lll=model.similarity('theatre','cinema')
#from nltk.corpus import wordnet
## w=wordnet.synsets('stripclub')[0]
## print(w.pos())
#w1=wordnet.synset('care.n.01')
#w2=wordnet.synset('hospital.n.01')
#print(w1.wup_similarity(w2))

#from gensim.models import word2vec
#sentences = word2vec.Text8Corpus('text8')
#model = word2vec.Word2Vec(sentences, size=200)
#
#model.most_similar(positive=['woman', 'king'], negative=['man'], topn=2)



#
#import pandas as pd
#df=pd.read_csv('export1.csv')
#amenity=df['amenity'].unique()
##
#from gensim.models import Word2Vec
#from sklearn.decomposition import PCA
#from matplotlib import pyplot
## define training data
##sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
##			['this', 'is', 'the', 'second', 'sentence'],
##			['yet', 'another', 'sentence'],
##			['one', 'more', 'sentence'],
##			['and', 'the', 'final', 'sentence']]
#sentences=[[x] for x in amenity][0:20]
##sentences=amenity
## train model
#model = Word2Vec(sentences, min_count=1)
## fit a 2d PCA model to the vectors
#X = model[model.wv.vocab]
#pca = PCA(n_components=2)
#result = pca.fit_transform(X)
## create a scatter plot of the projection
#pyplot.scatter(result[:, 0], result[:, 1])
#words = list(model.wv.vocab)
#for i, word in enumerate(words):
#	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
#pyplot.show()
#


#
#from gensim.scripts.glove2word2vec import glove2word2vec
#glove_input_file = 'glove.840B.300d.txt'
#word2vec_output_file = 'glove.840B.300d.txt.word2vec'
#glove2word2vec(glove_input_file, word2vec_output_file)
#

#
#from gensim.models import KeyedVectors
## load the Stanford GloVe model
#filename = 'glove.840B.300d.txt.word2vec'
#model = KeyedVectors.load_word2vec_format(filename, binary=False)
## calculate: (king - man) + woman = ?
#result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
#print(result)
#
