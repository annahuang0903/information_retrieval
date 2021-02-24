import pandas as pd
import gzip
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn

import re
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

import whoosh.index as index
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F

from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping


querylist = ["fast and cheap computer",
             "creamy bold red lipstick",
             "iconic eyeshadow palette",
             "iphone case with dog patten",
             "high dynamic range wide screen TV",
             "light and durable flashlight",
             "strong and healthy energy drink",
             "high definition Ebook with stylus",
             "bluetooth speaker that supports multiple devices",
             "fishing rod",
             "large and light duffel bag on wheel",
             "low-fat chocolate ice cream",
             "gentle and effective cleaning mask",
             "old-fashioned shirt",
             "Rainproof duffel bag with wheels",
             "luxury moisturizer for autumn",
             "low-fat high-protein chocolate ice cream",
             "white clothing hanger",
             "sharp killing knife",
             "human-form automatic calculator",
             "cheap and high-resolution TV",
             "high quality noise-cancelling headphone",
             "moisturizer for winter"]

beautyList = ["creamy bold red lipstick",
              "iconic eyeshadow palette",
              "gentle and effective cleaning mask",
              "luxury moisturizer for autumn",
              "lipstick by designer Rihanna",
              "pharmacists-recommended advanced scar gel",
              "body brush for dry skin",
              "daily moisturizing lotion with mild hyaluronic acid",
              "fragrance-free lotion for sensitive skin",
              "non-oily sleeping masks",
              "mild after-sun repair gel",
              "ultra lubricant eye-drops for dry eye symptom relief",
              "beautiful electric shaver for women",
              "Professional stainless steel tweezers",
              "new released rechargeable electric toothbrush",
              "gentle natural vitamin C serum for face",
              "liquid hand soap with lavender scent",
              "Pigmented matte eyeshadow palette with many colors",
              "hair dryer that does not hurt hair",
              "perfume for men with scent that attracts women",
              "cute water-proof makeup bags for travel",
              "concealer that is good",
              "water floss that protects plaque",
              "reuseable and disposable makeup remover"]

electronicsList = ["cheap and high resolution camera", "high quality noise-cancelling headphone",
                   ""]

#descriptionIndicators = set(["with", "which", "that"])
descriptionIndicators = set(["WDT", "WP", "WRB", "IN"])
possibleAdj = set(["JJ", "JJR", "JJS", "VBG", "VBN", "RB"])
possibleNoun = set(["NN", "NNS", "NNP", "NNPS"])


def query_syntactic_parser(query, descriptionIndicators, possibleAdj, possibleNoun):
    query = query.lower()
    hasDesIndicator = False
    desIndexSet = False
    blob = TextBlob(query)
    sentence = blob.sentences[0]
    #tokens = nltk.word_tokenize(query)
    #tagged = nltk.pos_tag(tokens)
    nounList = []
    adjList = []
    phaseList = blob.noun_phrases
    nounCounter = 0
    wordCounter = 0
    thatIndex = 0
    desIndex = nounCounter
    print(sentence.tags)
    for word, pos in sentence.tags:
        wordCounter += 1
        #if (word in descriptionIndicators) or pos == "IN":
        if pos in descriptionIndicators:
            hasDesIndicator = True
            if desIndexSet == False:
                desIndex = nounCounter
                desIndexSet = True
            if word == "that":
                thatIndex = wordCounter
        elif pos in possibleNoun:
            if hasDesIndicator:
                adjList.append(word)
            else:
                nounList.append(word)
            nounCounter += 1
        elif pos in possibleAdj:
            adjList.append(word)
    if hasDesIndicator:
        finalNounList = nounList[desIndex-1:desIndex]
    else:
        finalNounList = nounList[-1:]

    if len(finalNounList) != 0:
        for noun in finalNounList:
            if (len(phaseList)!= 0):
                for phase in phaseList:
                    if noun in phase:
                        #print(noun)
                        #print(phase)
                        phase = phase.split()
                        product = " ".join([i for i in phase if i not in adjList])
                        #print(adjList)
                        #print(product)
                    else:
                        product = noun
                    break
            else:
                product = noun
    else:
        product = query
    adjList.extend(nounList[:-1])
    if thatIndex != 0:
        afterthatDes = query.split(" ")[thatIndex:]
        adjList.extend(afterthatDes)
    finalAdjList = list(set([i for i in adjList if i not in product]))

    print("Noun: ", nounList)
    print("Final Noun List: ", finalNounList)
    print("Noun Phrase: ", phaseList)
    print("Adj: ", finalAdjList)
    print("Product:", product)
    print()
    
    return product, finalAdjList

#read in data and build data structure
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')



def makeDf(ins_dense, ins_meta, filterTime = None):
#    ins_meta=ins_meta[['asin','categories','title','description','salesRank']]
    ins_dense=ins_dense[['asin','summary','reviewText', 'unixReviewTime']]
    ins_dense_filt = ins_dense[ins_dense['unixReviewTime'] >= filterTime]
    ins_dense_filt['review']=ins_dense_filt['summary']+' '+ins_dense_filt['reviewText']
    ins_dense_filt=ins_dense_filt[['asin','review']]
    ins_review=ins_dense_filt.groupby('asin')['review'].apply(lambda x: ' '.join(x))
    ins_review=pd.Series.to_frame(ins_review)
    ins_review['asin']=ins_review.index
    ins=pd.merge(ins_meta,ins_review,how='right',on='asin')
    
    #process title
    ins['title']=ins['title'].str.lower().replace("(\d+(\.\d+)?)", "", regex=True).replace(r'[^\w\s]', "",regex=True)
    
    #process description
    ins['description']=ins['description'].str.lower().replace("(\d+(\.\d+)?)", "", regex=True).replace(r'[^\w\s]', "",regex=True)
    
    # make a new column concatenating title and description
    ins['TitleDesc'] = ins['title'] + ins['description']
    ins = ins.fillna('')
    return ins



def indexing(IdxDir, Df):
    schema =  Schema(name=ID(stored=True), text=TEXT(stored=False))
    ix = index.create_in(IdxDir, schema)
    writer = ix.writer()
    for i in range(len(Df)):
        name = Df['asin'][i]
        title = Df['title'][i]
        description = Df['description'][i]
        review = Df['review'][i]
        total = title + " " + description +" "+ review
        writer.add_document(name=name, text=total)
    writer.commit()
    
    return ix



#searching
def searching(idx, query, limit = 10):
    bm = BM25F()
    searcher = idx.searcher(weighting=bm)
    qp = QueryParser("text", schema=idx.schema)
    q = qp.parse(query)
    results = searcher.search(q, limit=limit)
    return results

def descriptionExpansion(adjList):
    expandedList = []
    tagToCheck = ["a", "s"]
    for adj in adjList:
        type = wn.synsets(adj)[0].pos()
        if type in tagToCheck:
            for syn in wn.synsets(adj):
                for name in syn.lemma_names():
                    name = name.lower()
                    if name not in expandedList:
                        expandedList.append(name)
    retList = list(set(adjList) | set(expandedList))
    return retList

def adjExpansion(adjList, expNum):
    expandedList = []
    for adj in adjList:
        iExpand = []
        for syn in wn.synsets(adj): 
            for l in syn.lemmas():
                iExpand.append(l.name())
        iExpand = iExpand[:expNum]
        expandedList.extend(iExpand)
        
    retList = list(set(adjList) | set(expandedList))
    return retList

def LSTM_setup():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,256,input_length=max_len)(inputs)
    layer = LSTM(128)(layer)
    layer = Dense(128,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(128,name='FC2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(156,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


############################ M A I N ############################


############## Get Df
reviewGz = "/Users/mikesung/Documents/SemanticQueryProject/SemanticQueryData/reviews_Electronics.json.gz"
metaGz = "/Users/mikesung/Documents/SemanticQueryProject/SemanticQueryData/meta_Electronics.json.gz"
reviewDf = getDF(reviewGz)
metaDf = getDF(metaGz)
reviewFilter = datetime.datetime(2012,1,1)
filterTime = time.mktime(reviewFilter.timetuple())
Df = makeDf(reviewDf, metaDf, filterTime = filterTime)
Df['TitleDesc'] = Df['title'] + Df['description']

############## Use Deep Learning to learn title-category mapping
category=[]
for i in range(len(Df)):
  cats=[]
  for j in range(len(Df['categories'][i][0])):
    if j>2:
      break
    else:
      cats.append(Df['categories'][i][0][j])
  cats = ",".join(cats)
  category.append(cats)
Df['reduced_cat'] = category
Df.reduced_cat = pd.Categorical(Df.reduced_cat)
Df['label'] = Df.reduced_cat.cat.codes
labelDf = Df[['label', 'reduced_cat']].drop_duplicates().sort_values('label')
labelDf = labelDf.reset_index(drop=True)

X_train,X_test,Y_train,Y_test = train_test_split(Df['title'],Df['label'],test_size=0.05)

max_words = 5000
max_len = 30
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

model = LSTM_setup()
model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

history = model.fit(sequences_matrix,Y_train,batch_size=512,epochs=20,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

# Calculate testing accuracy and loss
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
acc = model.evaluate(test_sequences_matrix,Y_test)

# Plot training and CV loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

############## Indexing

IdxDir = '/Users/mikesung/Documents/SemanticQueryProject/Amazon_Electronics_Filtered_Index'

# make new index

#idx = indexing(IdxDir, Df)

# open existing index

idx = index.open_dir(IdxDir)

############## Searching

userQ = "rgb gaming keyboard with low-latency"
product, adjList = query_syntactic_parser(userQ, descriptionIndicators, possibleAdj, possibleNoun)
expAdjList = adjExpansion(adjList, 3)
adjString = " OR ".join(expAdjList)
adjString = "(" + adjString + ")"
query = product + " AND " + adjString

numProduct = 1000
results = searching(idx, query, limit=numProduct)


############## Filter unwanted categories
# Get Model Prediction for categories
query_sequence = tok.texts_to_sequences([userQ])
query_sequences_matrix = sequence.pad_sequences(query_sequence,maxlen=max_len)
qPred = model.predict(query_sequences_matrix)

# Make top prediction dataframe with probability category information
qPredSort = np.sort(qPred,-1)
qPredSort = qPredSort.reshape(qPredSort.shape[1], 1)
qPredSort = qPredSort[::-1]

topLabels = np.argsort(qPred)
topLabels = topLabels.reshape(topLabels.shape[1], 1)
topLabels = topLabels[::-1]

topCategories = []
for i in range(len(topLabels)):
    topCategories.append(labelDf['reduced_cat'][topLabels[i][0]])
topCategories = np.array(topCategories).reshape(len(topCategories), 1)

topLabelProbCat = pd.DataFrame(np.concatenate((topLabels, qPredSort, topCategories), 1))


# Set how many categories to keep
retrieveTopCat = 1
topLabels = topLabels.reshape(len(topLabels), )
topLabels = topLabels[:retrieveTopCat]

retProductDf = pd.DataFrame()
if len(results) >= numProduct:
    for i in range(numProduct):
        productPost = Df[Df.asin == results[i]['name']]
        if productPost['label'] in topLabels:
            retProductDf = retProductDf.append(productPost)
else:
    print("Results are fewer than requested number of products, please modify and try again...")


