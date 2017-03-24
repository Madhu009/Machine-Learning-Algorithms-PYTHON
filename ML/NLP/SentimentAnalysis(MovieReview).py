import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

#1.Read the data file

train=pd.read_csv('C:/Users/Madhu/Desktop/NLP/train.tsv',header=0,
                  delimiter="\t",quoting=3)
test=pd.read_csv('C:/Users/Madhu/Desktop/NLP/test.tsv',header=0,delimiter="\t",
                 quoting=3)
print(train["SentenceId"])

'''
def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return (words)


#2.Clean the dataset remove unnessary words

clean=[]
for i in range(0,len(train["review"])):
    clean.append(" ".join(review_to_wordlist(train["review"][i],True)))


#3.creating bag of words

vector=CountVectorizer(analyzer="word",
                       tokenizer=None,
                       preprocessor=None,
                       stop_words=None,
                       max_features=5000)

featueVectors=vector.fit_transform(clean).toarray()

#4.Train the classifier

forest=RandomForestClassifier(n_estimators=100)
forest=forest.fit(featueVectors,train["sentiment"])


#5.testing

cleanTest=[]
for i in range(0,len(train["review"])):
    cleanTest.append(" ".join(review_to_wordlist(train["review"][i],True)))

featueVectorstest=vector.fit_transform(cleanTest).toarray()


#6.predict the input

result=forest.predict(featueVectorstest)
output=pd.DataFrame(data={"id":test["id"],"sentiment":result})

print(output)

'''