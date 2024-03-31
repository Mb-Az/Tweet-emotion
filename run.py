from template import NaiveBayesClassifier
import string
import pandas as pd
import os
import re
import nltk.tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import time

punctuation = string.punctuation
numbers = {'0','1','2','3','4','5','6','7','8','9'}
stopwords = set(stopwords.words('English'))

ps = PorterStemmer()

def has_this(word,this): #Why we ignore words with numbers? By this we would miss words like GPT4 or fly752 !
    for n in this:
        if n in word:
            return True
    return False

def preprocess(tweet_string):
    # cleaning the data and tokenizing it
    tweet_string = str(tweet_string)
    sentences = nltk.tokenize.sent_tokenize(tweet_string)
    features = []
    for sent in sentences:
        for word in nltk.word_tokenize(sent):
            if word.lower() not in stopwords and not has_this(word, numbers) and not has_this(word, punctuation):   
                features.append(ps.stem(word.lower()))
    return features

def load_data(data_path):
    df = pd.read_csv(data_path,encoding='utf-8')
    data = []
    for tweet_text,label in zip(df['text'],df['label']):
        data.append((preprocess(tweet_text),label))
    return data


# train your model and report the duration time
train_started_time = time.time()

here = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(here, "train_data.csv")

classes = ['negative', 'neutral','positive']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))

train_ended_time = time.time()

#Checking eval data and labeling new ones:
labeling_stated_time = time.time()

eval_data_path = os.path.join(here, "eval_data.csv")
eval_data = pd.read_csv(eval_data_path)

our_model = ""
hit_count = 0

index = 0
for text in eval_data["text"]:
    pred = nb_classifier.classical_classify(preprocess(str(text)))
    our_model += pred + "\n"

    if(eval_data.iloc[index, 4] == pred):
        hit_count += 1
    index += 1

labeling_ended_time = time.time()

#reporting hit rate and measured time for train and labeling:
print("hit_rate = ", round((hit_count/len(eval_data)*100),2) ,"Percent")
print("train duration = ", train_ended_time - train_started_time, " Seconds")
print("labeling duration = ", labeling_ended_time - labeling_stated_time, " Seconds")


#Write our model results fo file:
result_path = os.path.join(here, "result.txt")
f = open(result_path, 'w')
f.write(our_model)
f.close()