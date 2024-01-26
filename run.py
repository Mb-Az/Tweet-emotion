from template import NaiveBayesClassifier
import string
import pandas as pd
import os
import nltk.tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

punctuation = string.punctuation
numbers = {'0','1','2','3','4','5','6','7','8','9'}
alphabets = tuple(string.ascii_letters)
stopwords = set(stopwords.words('English'))

ps = PorterStemmer()

def has_numbers(word): #Why we ignor words with numbers? By this we would miss words like GPT4 or fly752 !
    for n in numbers:
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
            if word not in stopwords and word not in punctuation and not has_numbers(word) and word.startswith(alphabets): #should also clean I,you, my ...?
                features.append(ps.stem(word))
    return features

def load_data(data_path):
    df = pd.read_csv(data_path,encoding='utf-8')
    data = []
    for tweet_text,label in zip(df['text'],df['label']):
        data.append((preprocess(tweet_text),label))
    return data


# train your model and report the duration time
here = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(here, "train_data.csv")

classes = ['negative', 'neutral','positive']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))


#Checking eval data:
eval_data_path = os.path.join(here, "eval_data.csv")
eval_data = pd.read_csv(eval_data_path)


#eval_data["Our model"] = nb_classifier.classical_classify(preprocess(eval_data["text"]))
our_model = []
hit_count = 0

index = 0
for text in eval_data["text"]:
    pred = nb_classifier.classical_classify(preprocess(str(text)))
    our_model.append(pred)

    if(eval_data.iloc[index, 4] == pred):
        hit_count += 1
    index += 1

print("hit_rate = ", round((hit_count/len(eval_data)*100),2) ,"Percent")
