
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


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text

def remove_whitespace(text):
    text = re.sub(" +", " ", text)
    return text

def remove_newline(text):
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    return text

def clean_everything(text):
    text1 = remove_newline(text)
    text2 = remove_punctuations(text1)
    text3 = remove_whitespace(text2)
    return text3

# def get_str_from_tika(file_loc):
#     print("Tika Called")  # Tested on .pdf, .doc, .docx, .txt
#     raw = parser.from_file(file_loc)  # Javascript UI should restrict to these 4 filetypes
#     return raw["content"]

def my_tokeniser(text):
    mystr = ''
    for i in text:
        mystr += i
    sentences = sent_tokenize(mystr)
    return sentences

# def get_clean_strls_from_file(file_loc):
#     mysentences = my_tokeniser(get_str_from_tika(file_loc))
#     new_sentences = list(map(clean_everything, mysentences))
#     return new_sentences
def get_clean_strls_from_str(str):
    mysentences = my_tokeniser(str)
    new_sentences = list(map(clean_everything, mysentences))
    return new_sentences