from sumeval.metrics.rouge import RougeCalculator
import MeCab
import re
import os
import urllib.request
from collections import Counter
import pandas as pd
import numpy as np
from Mecab2 import Mecab2, Regexp



def evaluate_rouge_n(summary, references, n=2):
    rouge = RougeCalculator(stopwords=True, lang="ja")
    if type(n)==type(1):
        return rouge.rouge_n(summary=summary,
                             references=references,
                             n=n)
    else:
        if n == "l":
            return rouge.rouge_l(summary=summary,
                             references=references)
        elif n == "be":
            return rouge.rouge_be(summary=summary,
                             references=references)
        else:
            pass


def word_tokenize(doc):
    tagger = MeCab.Tagger("-Owakati")
    doc = str(doc)
    result = tagger.parse(doc)
    result = result.replace(" \n", "")
    list_of_tokens = result.split(" ")
    return list_of_tokens


def download_stopwords(path):
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    if os.path.exists(path):
        print('File already exists.')
    else:
        print('Downloading...')
        # Download the file from `url` and save it locally under `file_name`:
        urllib.request.urlretrieve(url, path)

def create_stopwords(path):
    stop_words = []
    for w in open(path, "r"):
        w = w.replace('\n','')
        if len(w) > 0:
          stop_words.append(w)
    return stop_words


def load_stopwords():
    path = "stop_words.txt"
    download_stopwords(path)
    stop_words = create_stopwords(path)
    return stop_words


def count_words(df):
    result = []
    for text in df["abstract"].values.tolist():
        text = str(text)
        tokens = word_tokenize(text)
        result.extend(tokens)
    
    for text in df["reference"].values.tolist():
        text = str(text)
        tokens = word_tokenize(text)
        result.extend(tokens)
    
    return Counter(result)


def get_stats(df):
    count = count_words(df)
    stat = {"word":list(count.keys()),
        "num": list(count.values())}
    stat = pd.DataFrame.from_dict(stat)
    num_tokens = 0
    for num in stat["num"]:
        num_tokens += int(num)

    rate = []
    for num in stat["num"]:
        rate.append(int(num)/num_tokens)
    
    stat["rate"] = rate
    return stat


def format_text(text):
    '''
    MeCabに入れる前のツイートの整形方法例
    '''

    text=re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    text=re.sub(r'[!-~]', "", text)#半角記号,数字,英字
    text=re.sub(r'[︰-＠]', "", text)#全角記号
    text=re.sub('\n', " ", text)#改行文字
    text=re.sub(r"[ａ-ｚ]", "", text) #全角英字
    return text


def normalize_and_removestop(text):
    c = Regexp()
    text = str(text)
    text = format_text(text)
    text1 = c.normalize(text)#ex) "南アルプスの天然水-Ｓｐａｒｋｉｎｇ*Ｌｅｍｏｎ+レモン一絞り"
    #print(text1) # 南アルプスの天然水-Sparking*Lemon+レモン一絞り
    m = Mecab2(target=["名詞","動詞","形容詞","副詞"])
    text2 = m.wakati(text1)
    #print(text2) # 南アルプスの天然水 Sparking Lemon レモン 一 絞る
    text3 = m.removeStoplist(text2, [])
    #print(text3) # 南アルプスの天然水 Sparking Lemon レモン 絞る
    return str(text3).split(" ")


def divide(text):
    text = format_text(text)
    tokens = normalize_and_removestop(text)
    length = len(tokens)
    sep = length//4
    output = []
    output.append(tokens[:sep])
    output.append(tokens[sep:sep*2])
    output.append(tokens[sep*2:sep*3])
    output.append(tokens[sep*3:])
    return output


def get_unique_unigram_rate(df):
    result_dict = {0:[],1:[],2:[],3:[]}
    for i in range(len(df)):
        abst_unigrams = normalize_and_removestop(df["abstract"][i])
        abst_unigrams = set(abst_unigrams)
        length = len(abst_unigrams)
        reference = df["reference"][i]
        segments = divide(reference)

        for n in range(4):
            seg = set(segments[n])
            count = 0
            for unigram in abst_unigrams:
                if unigram in seg:
                    count += 1
            rate = count/length
            result_dict[n].append(rate)
    for i in range(4):
        result_dict[i] = np.mean(result_dict[i])
    return result_dict


def get_ngram(text, n=2):
    text = str(text)
    text = format_text(text)
    tokens = word_tokenize(text)
    ngrams = []
    for i in range(len(tokens) - (n-1)):
        ngram = ""
        for q in range(n):
            ngram += tokens[i+q]
        ngrams.append(ngram)
    return ngrams


def load_dfs():
    ld = pd.read_excel("data/livedoor_refined.xlsx")
    latex = pd.read_excel("data/NLP_latex_refined.xlsx")
    patent = pd.read_excel("data/patent_refined.xlsx")
    return ld, latex, patent


def count_new_ngram_rate(df):
    result_dict = {1:[],2:[],3:[],4:[]}
    for i in range(len(df)):
        abstract = str(df["abstract"][i])
        reference = str(df["reference"][i])

        for n in range(1,5):
            ngrams = get_ngram(abstract, n=n)
            length = len(ngrams)
            count = 0
            if length:
                for ngram in ngrams:
                    if ngram not in reference:
                        count += 1
                result_dict[n].append(count/length)
    
    for i in range(1,5):
        result_dict[i] = np.mean(result_dict[i])
    return result_dict


def get_rouge_average(df):
    result_dict = {1 : [],
                    2 : [],
                    "l" : []}
    for i in range(len(df)):
        abst = format_text(df["abstract"][i])
        ref = format_text(df["reference"][i])
        for N in [1,2,"l"]:
            rouge = evaluate_rouge_n(abst, ref, n=N)
            result_dict[N].append(rouge)
    
    for N in [1,2,"l"]:
        result_dict[N] = np.mean(result_dict[N])
    
    return result_dict
