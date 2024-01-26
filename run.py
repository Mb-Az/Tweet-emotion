from template import NaiveBayesClassifier
import string
import pandas as pd
import nltk.tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

punctuation = string.punctuation
numbers = {'0','1','2','3','4','5','6','7','8','9'}
alphabets = tuple(string.ascii_letters)
stopwords = set(stopwords.words('English'))

ps = PorterStemmer()

def has_numbers(word):
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
train_data_path = 'train_data.csv'
classes = ['negative', 'neutral','positive']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))
print(nb_classifier.calculate_landa('mom',True))
test_string = "I love playing football"
print(nb_classifier.classify(preprocess(test_string)))
#what are eval_data and test_data_nolablel for?

