    def tfidf_test(self,testdat):
        results = dict()
        with open(testdat,'r') as fd:
            for line in fd.readlines():
                id,*words = line.split()
                print("=====")
                print(id)
                for word in words:
                    score = self.tfidf(word,id,words)
                    if id in results:
                        results[id].update({word:score})
                    else:
                        results.update({id:{word:score}})
                newA = dict(sorted(results[id].items(), key=operator.itemgetter(1), reverse=True)[:5])
                print(newA)

    def tf(self,word, id):
        '''return term frequency, number of times a word appears
        in a category normalized by dividing total num of words
        in the category'''
        return self.vocab[id][word]/len(self.vocab[id].values())

    def n_containing(self,word, idlist):
        '''Returns number of categories that contain word'''
        count = 0
        for id in self.classes.keys():
            if word in self.vocab[id].keys():
                count+=1
        return count
    def idf(self,word, idlist):
        '''Computes inverse document frequency measuring
        how common a word is among in all categories in training.txt
        More common a word is, lower its idf. Take ratio of total num
        of categories containing 'word', and take log of it. Add 1 to
        divisor'''
        return math.log(len(idlist))/(1+self.n_containing(word,idlist))

    def tfidf(self,word, id, idlist):
        '''Compute TF-IDF score, product of tf and idf'''
        return self.tf(word,id)*self.idf(word,idlist)
