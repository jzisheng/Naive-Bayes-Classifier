'''Loads test data and determines category of each test.  Assumes
train/test data with one text-document per line.  First item of each
line is category; remaining items are space-delimited words.  

Author: Zisheng Jason Chang

Date: 5.Nov.2017

'''
from __future__ import print_function
import sys
import math
import copy
import operator

class NaiveBayes():
    '''Naive Bayes classifier for text data.
    Assumes input text is one text sample per line.  
    First word is classification, a string.
    Remainder of line is space-delimited text.
    '''
    vocab = dict()          # Stores the vocabulary nested according to cat.
    categories = dict()        # Stores all the categories in the training document
    all_vocab = dict()         # Stores all vocab. regardless of class

    categoriesprob = dict()    # Stores the probabilities of each cat.
    vocabprob = dict()      # Stores the probabilities of each vocab word in cat.
    vocSize = 0
    def __init__(self,train):
        '''Create classifier using train, the name of an input
        training file.
        '''
        self.learn(train) # loads train data, fills prob. table

    ###################################
    # Function for learning data
    ###################################
    def learn(self,traindat):
        '''Load data for training; adding to 
        dictionary of classes and counting words.'''

        # Code below reads the training data and puts it all into
        # a dictionary
        with open(traindat,'r') as fd:
            for line in fd.readlines():
                category,*words = line.split()
                # Sees if dictionary key for 'post id' already exists,
                # If not add it, if it does exist access the key-value
                # pair and increment it
                self.categories[category] = self.categories.get(category,0)+1

                # Itereates through the words in the class id, and adds
                # the word to the key 'class id', and if it already exists
                # increment the word in the key 'class id' by 1.
                # For example...
                # {"alt.atheism":{"atheism":3,"archive":10, ...}, ... }
                for word in words:
                    # Collect and count total num of all unique words
                    self.all_vocab[word] = self.all_vocab.get(word,0)+1
                    if category not in self.vocab.keys():
                        self.vocab[category] = {word:1}
                    else:
                        self.vocab[category].update( {word:self.vocab[category].get(word,0)+1} )
        fd.close()

    ###################################
    # Launch appropiate testing version
    ###################################
    def test(self,testdat,version):
        if version ==  0:
            # Default test version
            self.raw_test(testdat)
        elif version ==  1:
            self.mest_test(testdat)
        elif version ==  2:
            self.tfidf_test(testdat)

    ###################################
    # Naive Bayes Raw Classification
    ###################################
    def raw_test(self,testdat):
        results = dict()
        correct = 0
        total = 0
        # Duplicate vocabulaies to store probabilities
        self.vocabprob = copy.deepcopy(self.vocab)  
        self.categoriesprob = copy.deepcopy(self.categories)   

        # Calculate total number of categories and size
        self.classTotal = len(self.categories.keys())  
        self.vocSize = len(self.all_vocab.values())
        #print("classTotal "+str(self.classTotal))
        #print("vocsize    "+str(self.vocSize))

        # Calculate probabilities
        for category in self.categories:
            # Calculate the probability of a P(Category)
            self.categoriesprob[category] = self.categories.get(category,1)/sum(self.categories.values())
            for word in self.vocab[category]:
                # Calculate P(word|category)
                num_word_in_cat = self.vocab[category][word]
                num_tot_words_in_cat = sum(self.vocab[category].values())
                #print("P("+word+"|"+category+") = "+str(num_word_in_cat)+"/"+str(num_tot_words_in_cat))
                self.vocabprob[category][word] = num_word_in_cat/num_tot_words_in_cat

        # Counter to test number of correct, and total
        correct = 0
        total = 0

        # Dict to store results
        results = dict()
        # Now open file, and test trained data
        with open(testdat,'r') as fd:
            for line in fd.readlines():
                total += 1 # Increment counter by 1
                r_category, *words = line.split() # split testing into category and words
                # Iterate through categories of vocabulary
                for category in self.categories:
                    # Fetch the P(C)
                    prob_category = self.categoriesprob[category]
                    prob_category_word = prob_category
                    # Now calculate P(C|W1,W2,W3...) = P(C)P(W1|C)P(W2|C)P(W3|C)...
                    for word in words:
                        # Gets the probability of word in category, otherwise returns 0
                        prob_category_word *= self.vocabprob[category].get(word,0)
                    #print("P("+category+"|"+word+") = "+str(prob_category_word))
                    results.update({category:prob_category_word})
                result = max(results, key=results.get)  # Get the NB Assumption of max Prob.
                if result == r_category:
                    correct += 1
        print("correct "+str(correct))
        print("total   "+str(total))
        print("Percent Correct: "+str(round((correct/total)*100,2))+"%")
        print("=====")

    ###################################
    # Function for m-estimation of class
    ###################################
    def mest_test(self,testdat):
        results = dict()
        correct = 0
        total = 0
        # Duplicate vocabulaies to store probabilities
        self.vocabprob = copy.deepcopy(self.vocab)  
        self.categoriesprob = copy.deepcopy(self.categories)   

        # Calculate total number of categories and size
        self.classTotal = len(self.categories.keys())  
        self.vocSize = len(self.all_vocab.values())
        #print("classTotal "+str(self.classTotal))
        #print("vocsize    "+str(self.vocSize))

        # Calculate probabilities
        for category in self.categories:
            # Calculate the probability of a P(Category)
            self.categoriesprob[category] = self.categories.get(category,1)/(sum(self.categories.values()))
            for word in self.vocab[category]:
                # Calculate P(word|category)
                num_word_in_cat = 1+self.vocab[category][word]
                num_tot_words_in_cat = sum(self.vocab[category].values())+self.vocSize
                #print("P("+word+"|"+category+") = "+str(num_word_in_cat)+"/"+str(num_tot_words_in_cat))
                self.vocabprob[category][word] = num_word_in_cat/num_tot_words_in_cat

        # Counter to test number of correct, and total
        correct = 0
        total = 0

        # Dict to store results
        results = dict()
        # Now open file, and test trained data
        with open(testdat,'r') as fd:
            for line in fd.readlines():
                total += 1 # Increment counter by 1
                r_category, *words = line.split() # split testing into category and words
                # Iterate through categories of vocabulary
                for category in self.categories:
                    # Fetch the P(C)
                    prob_category = self.categoriesprob[category]
                    # Number of words in category + vocabulary size
                    denom_mest = self.vocSize+sum(self.vocab[category].values())
                    # Being calculation of P(C)
                    prob_category_word = prob_category
                    # Now calculate P(C|W1,W2,W3...) = P(C)P(W1|C)P(W2|C)P(W3|C)...
                    for word in words:
                        prob_category_word *= self.vocabprob[category].get(word,1/denom_mest)
                    #print("P("+category+"|"+word+") = "+str(prob_category_word))
                    results.update({category:prob_category_word})
                result = max(results, key=results.get)  # Get the NB Assumption of max Prob.
                if result == r_category:
                    correct += 1
        print("correct "+str(correct))
        print("total   "+str(total))
        print("Percent Correct: "+str(round((correct/total)*100,2))+"%")
        print("=====")

    ###################################
    # Function(s) for tfidf estimation
    ###################################
    def tfidf_test(self,testdat):
        results = dict()
        correct = 0
        total = 0
        # Duplicate vocabulaies to store probabilities
        self.vocabprob = copy.deepcopy(self.vocab)  
        self.categoriesprob = copy.deepcopy(self.categories)   

        # Calculate total number of categories and size
        self.classTotal = len(self.categories.keys())  
        self.vocSize = len(self.all_vocab.values())
        #print("classTotal "+str(self.classTotal))
        #print("vocsize    "+str(self.vocSize))
        # Number of categories word is contained in
        n_cat_contain_voc = dict()

        # Calculate probabilities
        for category in self.categories:
            # Calculate the probability of a P(Category)
            self.categoriesprob[category] = self.categories.get(category,1)/sum(self.categories.values())
            for word in self.vocab[category]:
                # Calculate P(word|category)
                n_cat_contain_voc = {word:self.vocabprob.get(word,0)+1}
                num_word_in_cat = 1+self.vocab[category][word]
                num_tot_words_in_cat = sum(self.vocab[category].values())+self.vocSize
                #print("P("+word+"|"+category+") = "+str(num_word_in_cat)+"/"+str(num_tot_words_in_cat))
                self.vocabprob[category][word] = (num_word_in_cat/num_tot_words_in_cat)
        print(n_cat_contain_voc)
        
        # Counter to test number of correct, and total
        correct = 0
        total = 0
        # Dict to store results
        results = dict()
        
        # Counter for categories containing word in it
        # Open file, and test trained data
        with open(testdat,'r') as fd:
            for line in fd.readlines():
                total += 1 # Increment counter by 1
                r_category, *words = line.split() # split testing into category and words
                # Iterate through categories of vocabulary
                for category in self.categories:
                    # Fetch the P(C)
                    prob_category = self.categoriesprob[category]
                    prob_category_word = prob_category
                    # Number of words in category + vocabulary size
                    denom_mest = self.vocSize+sum(self.vocab[category].values())
                    # Now calculate P(C|W1,W2,W3...) = P(C)P(W1|C)P(W2|C)P(W3|C)...
                    for word in words:
                        if word in self.vocabprob[word]:
                            prob_category_word *= self.vocabprob[category][word]
                        else:
                            prob_category_word *= 1/sum(self.categories.values())+denom_mest
                    #print("P("+category+"|"+word+") = "+str(prob_category_word))
                    # Multiply by idf
                    
                    results.update({category:prob_category_word})
                result = max(results, key=results.get)  # Get the NB Assumption of max Prob.
                if result == r_category:
                    correct += 1
        print("correct "+str(correct))
        print("total   "+str(total))
        print("Percent Correct: "+str(round((correct/total)*100,2))+"%")
        print("=====")
    


def argmax(lst):
    return lst.index(max(lst))
    
def main():
    if len(sys.argv) != 4:
        print("Usage: %s trainfile testfile version" % sys.argv[0])
        sys.exit(-1)
    # Dict key-value for which version to use
    versions = {"raw":0,"mest":1,"tfidf":2}
    nbclassifier = NaiveBayes(sys.argv[1])
    #nbclassifier.test(sys.argv[2],versions.get(sys.argv[3],0))
    nbclassifier.test(sys.argv[2],versions.get(sys.argv[3],0))

if __name__ == "__main__":
    main()


    
