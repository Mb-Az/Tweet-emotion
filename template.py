# Naive Bayes 3-class Classifier 
# Authors: Baktash Ansari - Sina Zamani 

# complete each of the class methods  
import math
class NaiveBayesClassifier:

    def __init__(self, classes):
        # initialization: 
        # inputs: classes(list) --> list of label names
        # class_word_counts --> frequency dictionary for each class
        # class_counts --> number of instances of each class
        # vocab --> all unique words
        self.smoothing = 1  
        self.classes = classes
        self.class_word_counts = dict()
        self.class_counts = [0 for _ in range(len(classes))]
        self.vocab = None

    def train(self, data):
        # training process:
        # inputs: data(list) --> each item of list is a tuple 
        # the first index of the tuple is a list of words and the second index is the label(negetive, neutral, or positive)

        for features, label in data:
            self.class_counts[label] += 1 
            for word in features:
                if word not in self.class_word_counts:
                    self.class_word_counts[word] = [0,0,0]
                    self.class_word_counts[word][label] = 1
                else:
                     self.class_word_counts[word][label] += 1


    #def calculate_prior_log(self):
    #    # calculate log prior
    #    return math.log2(self.class_counts[2]/self.class_counts[0])
                     
    def calculate_prior(self, label):
        # calculate log prior
        if(label == 0):
            return self.class_counts[0]/(self.class_counts[1] + self.class_counts[2])
        if(label == 1):
            return self.class_counts[1]/(self.class_counts[0] + self.class_counts[2])
        return self.class_counts[2]/(self.class_counts[0] + self.class_counts[1])

    def calculate_likelihood(self, word, label):
        # calculate likelihhood: P(word | label)
        return (self.class_word_counts[word][label] + self.smoothing)/(self.class_counts[label] + self.smoothing*len(self.class_word_counts[word])) #double check the formula
    
    #def calculate_landa(self,word, neutral_effect = False):
    #    pp = self.calculate_likelihood(word,2) #P(positive | word)
    #    pn = self.calculate_likelihood(word,0) #P(negetice | word)
    #    if neutral_effect:
    #        return math.log2((pp + self.calculate_likelihood(word,1))/(pn+self.calculate_likelihood(word,1)))
    #    else:
    #        return math.log2(pp/pn)

    #def classify_with_lambda(self, features):
        # predict the class
        #landa = self.calculate_prior()
        #for word in features:
            #landa += self.calculate_landa(word)

        #should we set a threshold for classes? what is hyperparameter to train?
        #if landa > 0:
        #    return self.classes[2]
        #elif landa < 0:
        #    return self.classes[0]
        #else:
        #    return self.classes[1]
    
    def classical_classify(self, features):
        probs = dict()

        for label in range(len(self.classes)):
            probs[label] = 1
            for word in features:
                probs[label] *= self.calculate_likelihood(word, label)
            probs[label] *= self.calculate_prior(label)

        if(probs[0] > probs[1]):
            if(probs[0] > probs[2]):
                return(self.classes[0])
            return self.classes[2]
        if(probs[1] > probs[2]):
            return(self.classes[1])
        return self.classes[2]

# Good luck :)
