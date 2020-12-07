from nltk.downloader import ErrorMessage
import numpy as np
import nltk
import string
from bs4 import BeautifulSoup
import html as ihtml
import re
from pandas import DataFrame
from tika import parser 
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


class bruh:
      
    def __init__(self,resume_folder,job_desc):
        nltk.download('punkt')

        self.resume_folder = resume_folder
        self.job_desc = job_desc
        self.queries = []
        self.embedder = ''

################################ Clean Files to Sentence List ########################################
    def remove_punctuations(self, text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ')
        return text

    def remove_whitespace(self, text):
        text = re.sub(" +", " ", text)
        return text

    def remove_newline(self, text):
        text = text.replace("\n", "")
        text = text.replace("\t", "")
        return text

    def clean_everything(self, text):
        text1 = self.remove_newline(text)
        text2 = self.remove_punctuations(text1)
        text3 = self.remove_whitespace(text2)
        return text3

    def get_str_from_tika(self, file_loc):
        print("Tika Called")  # Tested on .pdf, .doc, .docx, .txt
        raw = parser.from_file(file_loc)  # Javascript UI should restrict to these 4 filetypes
        return raw["content"]

    def my_tokeniser(self, text):
        mystr = ''
        for i in text:
            mystr += i
        sentences = sent_tokenize(mystr)
        return sentences

    def get_clean_strls_from_file(self, file_loc):
        mysentences = self.my_tokeniser(self.get_str_from_tika(file_loc))
        new_sentences = list(map(self.clean_everything, mysentences))
        return new_sentences

    def get_clean_strls_from_str(self,str):
        mysentences = self.my_tokeniser(str)
        new_sentences = list(map(self.clean_everything, mysentences))
        return new_sentences

    ################################### Vector Transform Shit #######################################
    def get_vectorls_from_file(self, file_loc):
        ls = self.get_clean_strls_from_file(file_loc)
        re_ls = []
        for i in ls:
            re_ls.append(self.embedder.encode(i))
        return re_ls

    def get_csv_from_folder(self, folder):
        onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

        ls = self.get_clean_strls_from_file(file_loc)
    #################################################################################################

    ######################################## Algos ##################################################
    def run_uwu(self):
        tp_dd = {}
        re_dd = {}
        score_ls = []

        self.queries = self.get_clean_strls_from_file(self.job_desc)
        
        mypath = self.resume_folder
        print("Onlyfiles")
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        print(onlyfiles)

        for f in onlyfiles:
            resume_loc = self.resume_folder + f
            score = self.get_max_score_uwu(r"{}".format(resume_loc))
            tp_dd[str(score)] = resume_loc
            score_ls.append(score)
        print(score_ls)
        score_ls.sort(reverse=True)
        #count = 0
        for i in score_ls:
            #count += 1
            re_dd[tp_dd[str(i)]] = i
            print(i)
        return re_dd

    def run_bert(self, bert_variant):
        # Query sentences:
        tp_dd = {}
        re_dd = {}
        score_ls = []
        print("instanciate sentence_transform")
        self.embedder = SentenceTransformer(bert_variant)
        self.queries = self.get_clean_strls_from_file(self.job_desc)
        self.query_embedding = self.get_query_embeddings(self.queries)
        mypath = self.resume_folder
        print("Onlyfiles")
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        """onlyfiles = []
        for file in os.listdir(mypath):
            if file.endswith((".txt", ".pdf", ".docx", ".doc")):
                onlyfiles.append(os.path.join(mypath, file))
        """
        print(onlyfiles)

        for f in onlyfiles:
            resume_loc = self.resume_folder  + f
            score = self.get_max_score_bert(resume_loc)
            tp_dd[str(score)] = resume_loc
            score_ls.append(score)
        #print(score_ls)
        score_ls.sort(reverse=True)
        #count = 0
        for i in score_ls:
            #count += 1
            re_dd[tp_dd[str(i)]] = i.item()
            #print(i)
            '''
            re_dd[str(count)] = {resume_loc: tp_dd[str(i)],
                                    score: i}
            '''
        return re_dd

    def get_query_embeddings(self, queries):
        query_embedding = []
        for i in queries:
            query_embedding.append(self.embedder.encode(i, convert_to_tensor=True))
        return query_embedding

    def get_max_score_bert(self, file_loc):
        try:
            corpus = self.get_clean_strls_from_file(file_loc)
            corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True)

            queries = self.queries

            score_total = 0
            score_normaliser = len(queries)
            query_embedding = self.query_embedding
            for i in query_embedding:
                # query_embedding = self.embedder.encode(query, convert_to_tensor=True)
                cos_scores = util.pytorch_cos_sim(i, corpus_embeddings)[0]
                cos_scores = cos_scores.cpu()
                score_total += max(cos_scores)
            return score_total / score_normaliser
        except:
            print(file_loc + '\n file type not readable')
            pass

    def get_max_score_uwu(self, file_loc):
        try:
            corpus = self.get_clean_strls_from_file(file_loc)
            queries = self.queries
            score_total = 0
            score_normaliser = len(queries)
            for i in queries:
                max_score = 0
                #print("lmao")
                for j in corpus:
                    current_score = fuzz.ratio(i,j)
                    #print(i,j)
                    #print(current_score)
                    if max_score < current_score:
                        max_score = current_score
                    else:
                        pass
                    score_total += max_score
            return (score_total / score_normaliser) / 100
        except:
            print(file_loc + '\n file type not readable')
            pass

breh = bruh(r'/content/drive/MyDrive/Original_Resumes/', r'/content/drive/MyDrive/Job_Description/oaktree.txt')
cock1 = breh.run_bert('roberta-large-nli-stsb-mean-tokens')
cock = breh.run_uwu()


print(cock)
print(cock1)
