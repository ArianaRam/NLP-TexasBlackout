# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:10:58 2021

@author: ariana.ramos
"""
#import twitter
import pickle
import numpy as np

from pattern.nl import sentiment as sentimentnl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from pattern.nl import attributive, predicative as attnl, predicativenl
from pattern.nl import parse as parsenl
from pattern.nl import tag as tagnl
from pattern.en import attributive, predicative
from pattern.en import parse
from pattern.en import tag
from pattern.en import sentiment
import numpy as np
import requests
import os
import json
import csv
import datetime
import dateutil.parser
import unicodedata
import time 
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.dates import DateFormatter
#set graph styles
sns.set(style="darkgrid")
sns.set_context("paper")

import collections 
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib
import nltk
#nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from collections import Counter

#nltk.download('wordnet')      #download if using this module for the first time
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from os import path
from PIL import Image
import time 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#nltk.download('stopwords')    #download if using this module for the first time

#For Gensim

import gensim
import string
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
import plotly.graph_objects as go

from gensim.test.utils import common_corpus, common_dictionary

from gensim.models import HdpModel

from gensim.models.ldamodel import LdaModel
import plotly
import plotly.express as px
import text2emotion as te
from nltk.stem import PorterStemmer


#%% Function to find a specific tweet in text 

def user(x, df): 
    find= df['username'].loc[df['text'].str.contains(x).any()]
    return find



#%%Build LDA function

def oneLDA(df, num_topics, num_words):
 
    dictionary= corpora.Dictionary(df['lemrem'])
    DT_matrix= [dictionary.doc2bow(doc) for doc in df['lemrem']]
    Lda_object= gensim.models.ldamodel.LdaModel
    lda_model_1= Lda_object(DT_matrix, num_topics=num_topics, id2word=dictionary)
    LDA_results= (lda_model_1.print_topics(num_topics=num_topics, num_words=num_words))
    show= lda_model_1.show_topics(num_topics=num_topics, num_words=num_words)
    topics= list([lda_model_1[DT_matrix[n]] for n in range(len(DT_matrix))])
    return LDA_results, lda_model_1, topics
#%% Funtion to remove undesired words from stoplist. 

def remstop(docs, mystoplist):
    lemrem=[]
    for n in range(len(docs)):
        lemrem.append([t for t in docs['lemma'].iloc[n] if t not in mystoplist])
    return lemrem




#%% Function to get documents in topic

def docs_in_topic(topics, prob, num):
    top1= []
    
    #extract topic 1:
    
        
    for n in range(len(topics)):
        for i in range(4):
            if topics.iloc[n][i] is not None:
               if (topics.iloc[n][i][0]==num and topics.iloc[n][i][1]>prob):
                   top1.append(ercotall.iloc[n])

    return top1
#

#%% construct list per topic 
def list_topics(model):
    topic_words=pd.DataFrame(columns= range(10))
    weights= pd.DataFrame(columns= range(10))
    word_weight=[]
    for n in range(10):
        word_weight.append(pd.DataFrame.from_records(model.show_topic(n,10), columns=["word", "weights"])) 
        topic_words[n]= (word_weight[n]['word'])
        weights[n]= (word_weight[n]['weights'])
    return topic_words, weights

#%% Visualize Data in a timeline per Week 

def timeline(data, freq):
    data['date']= pd.to_datetime(data['created_at_x'])
    count= data.groupby(pd.Grouper(key='date', freq= freq, dropna=True )).count()
    return count['name']

#%%#%% most retweeted

def retweet(df):
    dummy= df
    metrics= pd.DataFrame(dummy.public_metrics_x)
    metrics.reset_index(inplace= True)
    retweet= [metrics.public_metrics_x.iloc[n].get('retweet_count') for n in range(len(metrics))]
    dummy['retweet']= retweet
    sortd= dummy.sort_values(by='retweet', ascending= False)
    return sortd[0:200]

#%% calculate Sentiment for the Janis Fadner coefficient 
def sent(lemmas):
    sent= pd.DataFrame([list(sentiment(n)) for n in lemmas])
    pos= sent[0].loc[sent[0]>0]
    neg= sent[0].loc[sent[0]<=0]
    return sent, pos, neg

#%% Janis Fadner coefficient 
def JanisFadner(Pos, Neg, neut):
    #P is the number of positive articles
    #N is th enumber of negative articles
    P= Pos 
    N= Neg+neut
    #P= Pos
    V= P+N +neut
    if V*V==0:
        JF='none'
    if P>N:
        JF= (P*P - P*N)/(V*V)
    if P==N:
        JF=0
    else: 
        JF= (P*N-N*N)/(V*V)
    return JF

#%%
def JFweekly(db):
    Smartmetsent, posSM, negSM= sent(db['lemrem'])
    db['polarity']= Smartmetsent[0]
  
    pos=[]
    neg=[]
    neut=[]
    
    for n in range(len(db)):
        if db['polarity'][n]>0:
            pos.append(1)
            neg.append(0)
            neut.append(0)
        if db['polarity'][n]==0:
            pos.append(0)
            neg.append(0)
            neut.append(1)
        if db['polarity'][n]<0:
            pos.append(0)
            neg.append(1)
            neut.append(0)
        
    db['pos']= pos
    db['neg']=neg
    db['neut']= neut
    
    poscount= db.groupby(pd.Grouper(key='date', freq='w', dropna=True)).sum()
    
    
    JFweek=pd.DataFrame([JanisFadner(poscount['pos'][n],poscount['neg'][n],poscount['neut'][n]) for n in range(len(poscount))], index=poscount.index)
    
    return JFweek
#%% Open downloaded and reformatted Dataset

with open('ercotall', 'rb') as f:
    ercotall= pickle.load(f)


#%%

timeline= timeline(ercotall, 'W')



#%% Number of Tweets per week (Figure 3)
sns.color_palette('light:#5A9', as_cmap= True)
fig, ax= plt.subplots(figsize= (10,4))
myFmt = DateFormatter('%b')
matplotlib.rc('font', size=30)
matplotlib.rc('axes', titlesize=12, labelsize= 12)
matplotlib.rc('ytick', labelsize= 12)
matplotlib.rc('xtick', labelsize= 12)
ax.xaxis.set_major_formatter(myFmt); 
#sns.color_palette('light:#5A9', as_cmap= True)
#sns.barplot(x, y, color= 'pink')
plt.plot(timeline)
#plt.plot(smarttime)
#plt.xticks(rotation=45)
ax.set(title= 'Number of tweets per week: \'Ercot\'')
ax.set(xlabel= 'date')
ax.set(ylabel= 'Tweets')

#%% remove repeated entries 
un, ind, ct= np.unique(ercotall['lemma'], return_counts=True, return_index=True)


ercunique= ercotall.iloc[ind]

ercunique['oldindex']=ercunique.index


#%% Slice only Februray data-  erc unique february

ercotfeb= ercunique.loc[ercunique['date']< '2021-03-01T00:00:00.000Z']





#%%

joinuni= [" ".join(n) for n in ercunique['lemma']]


#%% Run emotional package: Text2emotion -- TAKES A LONG TIME TO RUN 


#topemouni= pd.DataFrame([te.get_emotion(n) for n in joinuni])


#%% Save variables  
#with open('emouni', 'wb') as f:
 #   pickle.dump(topemouni, f)

#%% Read data 

    
with open('emouni', 'rb') as f:
    topemouni= pickle.load(f)
    

#%%
topemouni.index= ercunique['date']

#%%
happyall= topemouni['Happy'].resample('D').mean().bfill()
angryall= topemouni['Angry'].resample('D').mean().bfill()
surpriseall=  topemouni['Surprise'].resample('D').mean().bfill()
sadall=  topemouni['Sad'].resample('D').mean().bfill()
fearall=  topemouni['Fear'].resample('D').mean().bfill()

#%% Plot emotional content in Tweets

fig, ax= plt.subplots(figsize= (10,4))
myFmt = DateFormatter('%b')
matplotlib.rc('font', size=30)
matplotlib.rc('axes', titlesize=12, labelsize= 12)
matplotlib.rc('ytick', labelsize= 12)
matplotlib.rc('xtick', labelsize= 12)

plt.plot(happyall)
plt.plot(angryall)
plt.plot(surpriseall)
plt.plot(sadall)
plt.plot(fearall)
ax.xaxis.set_major_formatter(myFmt);
ax.legend(['happy', 'angry', 'surprised', 'sad', 'afraid'])
ax.set_title('Emotional Content in Tweets')

#%% Select February only

topemouni['index']= ercunique.index
#%%
emofeb= topemouni.loc[topemouni.index< '2021-03-01T00:00:00.000Z']

#%%

happyfeb= emofeb['Happy'].resample('D').mean().bfill()
Angryfeb= emofeb['Angry'].resample('D').mean().bfill()
Surprisefeb= emofeb['Surprise'].resample('D').mean().bfill()
Sadfeb= emofeb['Sad'].resample('D').mean().bfill()
Fearfeb= emofeb['Fear'].resample('D').mean().bfill()

#%% Emotional content February (Figure 6)

fig, ax= plt.subplots(figsize= (10,4))
myFmt = DateFormatter('%d-%b')
matplotlib.rc('font', size=30)
matplotlib.rc('axes', titlesize=12, labelsize= 12)
matplotlib.rc('ytick', labelsize= 12)
matplotlib.rc('xtick', labelsize= 12)
ax.xaxis.set_major_formatter(myFmt);
plt.plot(happyfeb)
plt.plot(Angryfeb)
plt.plot(Surprisefeb)
plt.plot(Sadfeb)
plt.plot(Fearfeb)
ax.legend(['happy', 'angry', 'surprised', 'sad', 'afraid'])
ax.set_title('Emotional Content in Tweets')

#%% select subsets of data based on emotions and then on emotional peaks
# select emotions from set: 
#unique indicator in set
nique, ct= np.unique(ercunique.index, return_counts=True)

#%% Remove stopwords 

stop_words=  nltk.corpus.stopwords.words('english')
newstop=['ercot','new', 'http', 'na', 'tx','ga', 'people', 
         'said', 'energy', 
         'rt','u','amp', 'u', 'hi', 'via', 'power', 'texas', 'grid', 'texan']
stop_words.extend(newstop)


ercunique['lemrem']= ercunique['lemma'].apply(lambda x:[word for word in x if word not in stop_words])

#%% Process text- remove stems 
ps = PorterStemmer()
stem= ercunique['lemrem'].apply(lambda x: [ps.stem(w) for w in x]) 

ercunique['lemrem']= stem

#%%Make sets per emotion
happymask= [emofeb['index'][n] for n in range(len(emofeb)) if emofeb['Happy'][n]>0]

happyset= ercunique[ercunique.index.isin(happymask)]


angrymask=[emofeb['index'][n] for n in range(len(emofeb)) if emofeb['Angry'][n]>0]
angryset= ercunique[ercunique.index.isin(angrymask)]


surprisemask= [emofeb['index'][n] for n in range(len(emofeb)) if emofeb['Surprise'][n]>0]
surpriseset= ercunique[ercunique.index.isin(surprisemask)]

sadmask= [emofeb['index'][n] for n in range(len(emofeb)) if emofeb['Angry'][n]>0]
sadset= ercunique[ercunique.index.isin(sadmask)]

fearmask= [emofeb['index'][n] for n in range(len(emofeb)) if emofeb['Fear'][n]>0]
fearset= ercunique[ercunique.index.isin(fearmask)]

#%% RUN LDA on each (takes a few minutes)

docsangry, modelangry, angrytopics= oneLDA(angryset, 10, 10)
angrywords, angryweights= list_topics(modelangry)

docshappy, modelhappy, happytopics= oneLDA(happyset, 10, 10)

happywords, happyweights= list_topics(modelhappy)

 
docssurp, modelsurp, surptopics= oneLDA(surpriseset, 10, 10)
surpwords, surpweights= list_topics(modelsurp)



docssad, modelsad, sadtopics=oneLDA(sadset, 10, 10)
sadwords, sadweights= list_topics(modelsad)


docsfear, modelfear, feartopics= oneLDA(fearset, 10, 10)
fearwords, fewarweights= list_topics(modelfear)

#%% make sunburst 
parentwords= pd.Series(['happy', 'angry', 'surprised', 'sad', 'afraid'])
#%%
kidhappy=[", ".join(happywords.iloc[0:3,n]) for n in range(4)]

kidangry= [", ".join(angrywords.iloc[0:3,n]) for n in range(4)]
kidsurprised=[", ".join(surpwords.iloc[0:3,n]) for n in range(4)]
kidsad= [", ".join(sadwords.iloc[0:3,n]) for n in range(4)]
kidfear= [", ".join(fearwords.iloc[0:3,n]) for n in range(4)]

#%%

kids= kidhappy+ kidangry+kidsurprised+kidsad+kidfear
parentcol2= pd.DataFrame(parentwords.repeat(4))


#%% change value to % of total for each emotion
vals= pd.Series(happyweights.iloc[0,:])
s2=vals.append(angryweights.iloc[0,:])
s3= s2.append(surpweights.iloc[0,:])
s4= s3.append(sadweights.iloc[0,:])
s5= pd.DataFrame(s4.append(fewarweights.iloc[0,:]))
s5.reset_index(inplace=True)
#%% Note that because the LDA has ramdomness in it, the results may vary 

happymean= emofeb['Happy'].resample('m').mean().bfill()
angrymean= emofeb['Angry'].resample('m').mean().bfill()
surprisemean= emofeb['Surprise'].resample('m').mean().bfill()
sadmean= emofeb['Sad'].resample('m').mean().bfill()
fearmean= emofeb['Fear'].resample('m').mean().bfill()

values= pd.Series([happymean[0], angrymean[0], surprisemean[0], sadmean[0], fearmean[0]])
#%%
values2= pd.DataFrame(values.repeat(4))
#%%
parentcol2['kids']= kids

#parentcol2['val']=s5[0]
parentcol2['val']= values2

parentcol2=parentcol2.rename(columns={0:'parent'})

#%%
fig= px.sunburst(parentcol2, path=['parent', 'kids'], values='val', 
                 title='LDA Topics per Emotion',
                 )
fig.update_layout(
    title={
        'text': "LDA Topics per Emotion in Tweets",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    font= dict(
        family="Times New Roman, monospace",
        size=12,
        color="black"
    ))
#fig.update_traces(insidetextorientation='radial')
#fig.update_layout(uniformtext=dict(minsize=8, mode= 'show'))

fig.show()
fig.write_image("sunburst_chart.png")



#%% find the relevant tweets 
metrics= pd.DataFrame(ercotfeb['public_metrics_x'])
#metrics['index']= ercotfeb.index
metrics.reset_index(inplace=True)
#%%

liked= pd.DataFrame.from_records(metrics['public_metrics_x'], columns=['retweet_count', 'reply_count', 'like_count', 'quote_count'])

#%%
retweet= pd.DataFrame([metrics.public_metrics_x.iloc[n].get('retweet_count') for n in range(len(metrics))])
#%%
retweet['ind']= ercotfeb.index

#%%
metricshappy= pd.DataFrame(happyset.public_metrics_x)
metricshappy.reset_index(inplace=True)
metricshappy['retweet']= pd.DataFrame([metricshappy.public_metrics_x.iloc[n].get('retweet_count') for n in range(len(metricshappy))])

#%%
#max 
happypop= happyset.iloc[np.argmax(metricshappy['retweet'])]
happytext= happypop.text

#%% Find tweets with max happiness
maxhappy= ercunique.iloc[(np.argmax(emofeb['Happy']))]
maxangry= ercunique.iloc[(np.argmax(emofeb['Angry']))]
maxsurprise= ercunique.iloc[(np.argmax(emofeb['Surprise']))]
maxsad= ercunique.iloc[(np.argmax(emofeb['Sad']))]
maxfear= ercunique.iloc[(np.argmax(emofeb['Fear']))]

#%% Find tweet topics:
    
musk= happyset[happyset['text'].str.contains("musk")]



#%%Most retweeted in topic
muskret= retweet(musk)
#%% tweet topics
cruz= sadset[sadset['text'].str.contains('cruz')]
#%%
cruzret= retweet(cruz)

#%%
oncor= sadset[sadset['text'].str.contains("oncor")]
oncorret= retweet(oncor)

#%%
cpsenergy= sadset[sadset['text'].str.contains("cpsenergy")]
cpsret= retweet(cpsenergy)

#%% 
allretweet= retweet(ercotall)
#%%
wind= sadset[sadset['text'].str.contains("wind")]
windret= retweet(wind)

#%% 
hacker= ercotall[ercotall['text'].str.contains('hacker')]
hackret= retweet(hacker)

#%%
mostret= ercotall[ercotall['text'].str.contains('The blackouts in Texas are primarily because of frozen instruments at gas, coal and nuclear plants -- as well limited supplies of gas, according to Ercot. Frozen wind turbines were the least significant factor')]

#%%
#emotional content of most retweeted
mostretuni= retweet(ercunique)

#%%
emomostret=topemouni[topemouni['index']==20857]



#%%
sadretweet= retweet(sadset)
#%%
angryretweet= retweet(angryset)
#%%
surprisedtweet= retweet(surpriseset)
#%% Run LDA on topic subset: 
docsmusk, modelmusk, musktopics= oneLDA(musk, 10, 10)
muskwords, muskweights= list_topics(modelmusk)


#%% Sentiment on entire dataset

sentunique, posuni, neguni= sent(ercunique['lemrem'])

#%% 
ercunique['polarity']= sentunique[0]
#%%
ercunique2= ercunique
ercunique2.reset_index(inplace=True) 

#%%Arrive at a weekly estimation of the Janis Fadner Coefficient: 
jfweekuni= JFweekly(ercunique2)

#%% font settings:
    
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}
#%%Media tenor in terms of Janis-Fadner Coefficient for 2021 (Figure 5)

myFmt = DateFormatter('%b')
matplotlib.rc('font', size=30)
matplotlib.rc('axes', titlesize=12, labelsize= 12)
matplotlib.rc('ytick', labelsize= 12)
matplotlib.rc('xtick', labelsize= 12)

plt.rc('font', **font)
sns.color_palette('light:#5A9', as_cmap= True)
fig, ax= plt.subplots(figsize= (10,4))
plt.rcParams.update({'font.size': 34})
plt.plot(jfweekuni, color='mediumorchid', linewidth=1 )
ax.set(title= 'Media Tenor: Twitter')
ax.set(xlabel= 'date')
ax.xaxis.set_major_formatter(myFmt); 
ax.set(ylabel= 'JF coefficient')


