import pandas as pd
import nltk
import string
from bs4 import BeautifulSoup
import html as ihtml
import re
from pandas import DataFrame
#from tika import parser 
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import torch
import os
from os import listdir
from os.path import isfile, join
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
nltk.download('stopwords')
#from pyresparser import ResumeParser
import transformers
from transformers import AutoTokenizer, AutoModel
from clean_str import *


df = pd.read_csv('job_desc/Top30.csv',header = 0,names = ['crap0','crap1','Title','Description'])
jobs_df = df.drop(['crap0', 'crap1'], axis=1)
print(jobs_df.shape)
jobs_df = jobs_df.drop_duplicates(keep='first')
# print(jobs_df.shape)
# jobs_df = jobs_df.drop_duplicates(subset = ['Title'],keep='first')
print(jobs_df.shape)
jobs_df.head()

#takes string returns
bert_variant = 'roberta-large-nli-stsb-mean-tokens'
embedder = SentenceTransformer(bert_variant)
def BERTify(mystr): 

    return embedder.encode(mystr)

shitcum = BERTify('tim')#'distilbert-base-nli-stsb-mean-tokens')
print(type(shitcum))
print(shitcum.shape)

titles_ls = []
desc_ls = []

for i in range(len(jobs_df)):
  #titles
  title = jobs_df.iloc[i,0]
  t_vec = BERTify(title)
  titles_ls.append(t_vec)

  #desc to desc sent vectors
  desc = jobs_df.iloc[i,1]
  desc_sentences = get_clean_strls_from_str(desc)
  desc_sentences_vector = []
  for j in desc_sentences:
    desc_sentences_vector.append(BERTify(desc))
  desc_ls.append(desc_sentences_vector)

job_vectors_df = pd.DataFrame(list(zip(titles_ls, desc_ls)), columns=['Title_vector','Desc_vector'])
job_vectors_df.head()
job_vectors_df.to_csv()