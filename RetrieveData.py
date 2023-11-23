#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Sep 22 14:10:58 2021


@author:  Ariana Ramos Gutierrez 

linkedIn: https://www.linkedin.com/in/ariana-ramos-gutierrez/

web: www.arianaramos.com

MIT copyright license. 

#Note: this code was valid at the time of writing the research project and up to February 2022,
#Policy at Twitter has changed after being acquired and rebranded at X. 
#Access tokens and methods may have changed. 


"""
import twitter
import pickle
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
from nltk.tokenize import word_tokenize
from collections import Counter

#nltk.download('wordnet')      #download if using this module for the first time
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from os import path
from PIL import Image
import time 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

nltk.download('stopwords')    #download if using this module for the first time

#For Gensim

import gensim
import string
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
import plotly.graph_objects as go

from gensim.test.utils import common_corpus, common_dictionary

from gensim.models import HdpModel

from gensim.models import LdaModel
import plotly
import plotly.express as px
import text2emotion as te
#%%#%%


#define in envirionment (not needed after running it once)
os.environ['TOKEN']= 'YOUR TWITTER ACCESS TOKEN'

#%% Function to retrieve authenticaton TOKEN:
    
def auth():
    return os.getenv('TOKEN')
#%% CREATE HEADERS

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

#%% #%% Create url 

def create_url(keyword, start_date, end_date, max_results= 10):
    search_url= 'https://api.twitter.com/2/tweets/search/all'
    
    # change params based on endpoint I'm using
    
    query_params= { 'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)


#%%
def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()
#%%
def twus(json_response):
    tweets= pd.DataFrame(json_response['data'])
    users= pd.DataFrame(json_response['includes']['users'])
    tweet_user= tweets.merge(users, left_on= tweets['author_id'], right_on= users['id'])
    return tweet_user

#%%

def tweet1batch(keyword, start_time, end_time, max_results, next_token):
    ''' for up to 500 results ''' 
    bearer_token= auth()
    headers= create_headers(bearer_token)
    url= create_url(keyword, start_time, end_time, max_results)
    json_response1= connect_to_endpoint(url[0], headers, url[1], next_token)
    return json_response1
#%%
def tweetnolim(keyword, start_time, end_time, max_results, maxlim):
    count= 0
    max_count= maxlim
    response= tweet1batch(keyword, start_time, end_time, max_results,None)
    data= twus(response)
    #data=[]
   
    if 'next_token' in response['meta']:
       next_token= response['meta']['next_token'] 
       count+= response['meta']['result_count'] 
    else:
        next_token=None

    while next_token is not None: 
        if count <= max_count: 
            response= (tweet1batch(keyword, start_time, end_time, max_results, next_token))
            mylist= twus(response)
            data= data.append(mylist, ignore_index=True)
            count+= response['meta']['result_count']
            time.sleep(2)
                    
            if 'next_token' in response['meta']:
                next_token= response['meta']['next_token']
                print(count) #remove for speed
            else: 
                next_token= None
        else: 
            break 
            print('count limit reached')
    return data


#%% # search blackouts in texas

#bearer_token= auth()
#headers= create_headers(bearer_token)
keyword= 'ercot lang:en'
  #"point_radius:[50.85045 4.34878 25mi]"
start_time= '2021-02-01T00:00:00.000Z'
end_time= '2021-02-16T04:20:00.000Z'
max_results= 500
maxlim= 200000

#%% count tweets 


#%%
ercot, count= tweetnolim(keyword, start_time, end_time, max_results, maxlim)

#%%
ercot2= tweetnolim(keyword, start_time, end_time, max_results, maxlim)

#%% 
ercot3= tweetnolim(keyword, start_time, end_time, max_results, maxlim)
#%%
ercot4= tweetnolim(keyword, start_time, end_time, max_results, maxlim)

#%%
ercot5= tweetnolim(keyword, start_time, end_time, max_results, maxlim)

#%%
ercot6= tweetnolim(keyword, start_time, end_time, max_results, maxlim)

#%%
ercot7= tweetnolim(keyword, start_time, end_time, max_results, maxlim)

#%%

ercotall= ercot.append([ercot2,ercot3,ercot4,ercot5,ercot6,ercot7], ignore_index=True)


#%%SAVEVAR

with open('ercotall', 'wb') as f:
    pickle.dump(ercotall, f)
