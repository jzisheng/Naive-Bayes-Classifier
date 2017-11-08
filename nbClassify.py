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
    classes = dict()        # Stores all the categories in the training document
    avocab = dict()         # Stores all vocab. regardless of class

    classesprob = dict()    # Stores the probabilities of each cat.
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
                id,*words = line.split()
                # Sees if dictionary key for 'post id' already exists,
                # If not add it, if it does exist access the key-value
                # pair and increment it
                self.classes[id] = self.classes.get(id,0)+1

                # Itereates through the words in the class id, and adds
                # the word to the key 'class id', and if it already exists
                # increment the word in the key 'class id' by 1.
                # For example...
                # {"alt.atheism":{"atheism":3,"archive":10, ...}, ... }
                for word in words:
                    # Collect and count total num of all unique words
                    self.avocab[word] = self.avocab.get(word,0)+1
                    if id not in self.vocab.keys():
                        self.vocab[id] = {word:1}
                    else:
                        self.vocab[id].update( {word:self.vocab[id].get(word,0)+1} )
        print(self.classes)
        print(self.vocab)
        fd.close()

    ###################################
    # Launch testing
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
    # Function(s) for tfidf estimation
    ###################################
    def tfidf_test(self,testdat):
        results = dict()
        correct = 0
        total = 0
        # Duplicate vocabulary to new dictionary for vocab probabilities
        self.vocabprob = copy.deepcopy(self.vocab)  
        # Duplicate classes to new dictionary for class probabilities
        self.classesprob = self.classes.copy() 
        # Calculate total number of classes
        self.classTotal = len(self.classes.keys())  
        # Calculate total vocabulary size
        self.vocSize = len(self.avocab.values())      

        # Iterate through the classes in the vocabulary
        for id in self.vocab:
            # Calculate the P(Class), and store in ClassProb
            #self.classesprob[id] = self.classes[id]/self.classTotal

            # Compute total num. of vocab words in 'Class', denominator of mest
            n_vocab_in_id = sum(self.vocab[id].values())
            # Iterate through the words in the vocabulary dicitonary by 'Class'
            for word in self.vocab[id]:
                #print("P("+word+"|"+id+") = "+str(self.vocab[id][word]+1)+"/"+str(n_vocab_in_id))
                # No. of times word is found in postid
                n_word = self.vocab[id][word]
                # Number of occurences smoothed by 1
                p_word_in_cat = (n_word+1)/(n_vocab_in_id+self.vocSize)
                # Return the result, and multiply by porportion
                # (# of times seen word)/(# of times word)
                self.vocabprob[id][word]=math.log(p_word_in_cat*(self.classTotal/n_vocab_in_id+1))

        with open(testdat,'r') as fd:
            # Increment through lines in the training files
            for line in fd.readlines():
                total+=1    # Increment total test by 1
                id,*words = line.split()
                # Iterate through all classes P(idClass)
                for idClass in self.classes:
                    # Get probability of P(idClass)
                    nbProb = self.classesprob[idClass]
                    # Get dict that stores prob. of P(word|idClass)
                    _class = self.vocabprob[idClass]
                    n_vocab_in_id = sum(self.vocab[idClass].values())
                    # Get P(word|classID)
                    # Iterate through all the words in the training set
                    for word in words:
                        # Either get probability of P(word|class) or smooth out to
                        # 1/# of words in class
                        _vocabp = _class.get(word,(1/n_vocab_in_id))       
                        # Multiply P(classId)*P(word_n|idclass_n)
                        #print("P("+word+"|"+idClass+")" +" = "+str(_vocabp))
                        nbProb+=_vocabp
                    results.update({idClass:nbProb})
                    #print("idclass "+idClass+" :"+str(_vocabp))
                result = max(results, key=results.get)  # Get the NB Assumption
                if result == id:
                    correct += 1    # NB Guessed correctly, increment by 1
        #print(results)
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

        # Calculate total vocabulary size
        self.classesprob = self.classes.copy() 
        for _class in self.classes:
            self.classesprob[_class] = self.classes[_class]/sum(self.classes.values())

        self.vocSize = len(self.avocab.keys())     

        with open(testdat,'r') as fd:
            for line in fd.readlines():
                total += 1 # Increment counter by 1
                id, *words = line.split()

                for _id in self.vocab:
                    id_prob = self.classesprob[_id]
                    for word in words:
                        #print("id prob "+_id+":"+str(id_prob))
                        n_k_num = self.vocab[_id].get(word,0)+1
                        n_v_denom = sum(self.vocab[_id].values())+self.vocSize
                        #print("P("+word+"|"+_id+")="+str(n_k_num)+"/"+str(n_v_denom))
                        id_prob *= (n_k_num)/n_v_denom
                    results.update({_id:id_prob})
                result = max(results, key=results.get)  # Get the NB Assumption
                if result == id:
                    correct += 1    # NB Guessed correctly, increment by 1'''

        print("correct "+str(correct))
        print("total   "+str(total))
        print("Percent Correct: "+str(round((correct/total)*100,2))+"%")
        print("=====")

    ###################################
    # Naive Bayes Raw Classification
    ###################################
    def raw_test(self,testdat):
        results = dict()
        correct = 0
        total = 0
        # Duplicate vocabulary to new dictionary for vocab probabilities
        self.vocabprob = copy.deepcopy(self.vocab)  
        # Duplicate classes to new dictionary for class probabilities
        self.classesprob = self.classes.copy() 
        # Calculate total number of classes
        self.classTotal = len(self.classes.keys())  
        # Calculate total vocabulary size
        self.vocSize = len(self.avocab.values())      

        # Iterate through the classes in the vocabulary
        for id in self.vocab:
            # Calculate the P(Class), and store in ClassProb
            #self.classesprob[id] = self.classes[id]/self.classTotal

            # Compute total num. of vocab words in 'Class', denominator of mest
            n_vocab_in_id = sum(self.vocab[id].values())+self.vocSize
            # Iterate through the words in the vocabulary dicitonary by 'Class'
            for word in self.vocab[id]:
                #print("P("+word+"|"+id+") = "+str(self.vocab[id][word]+1)+"/"+str(n_vocab_in_id))
                # No. of times word is found in postid
                p_word = self.vocab[id][word]
                # Number of occurences smoothed by 1
                p_word_in_cat = (p_word+1)/((n_vocab_in_id))
                # Return the result
                self.vocabprob[id][word]=p_word_in_cat

        with open(testdat,'r') as fd:
            # Increment through lines in the training files
            for line in fd.readlines():
                total+=1    # Increment total test by 1
                id,*words = line.split()
                # Iterate through all classes P(idClass)
                for idClass in self.classes:
                    # Get probability of P(idClass)
                    nbProb = self.classesprob[idClass]
                    # Get dict that stores prob. of P(word|idClass)
                    _class = self.vocabprob[idClass]
                    n_vocab_in_id = sum(self.vocab[idClass].values())
                    # Get P(word|classID)
                    # Iterate through all the words in the training set
                    for word in words:
                        # Either get probability of P(word|class) or smooth out to
                        # 1/# of words in class
                        _vocabp = _class.get(word,(1/n_vocab_in_id))       
                        # Multiply P(classId)*P(word_n|idclass_n)

                        #print("P("+word+"|"+idClass+")" +" = "+str(_vocabp))
                        nbProb*=_vocabp
                    results.update({idClass:nbProb})
                    #print("idclass "+idClass+" :"+str(_vocabp))
                result = max(results, key=results.get)  # Get the NB Assumption
                if result == id:
                    correct += 1    # NB Guessed correctly, increment by 1
        #print(results)
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


    
