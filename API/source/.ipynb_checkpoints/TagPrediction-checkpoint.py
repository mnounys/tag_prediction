import pandas as pd
import os
import joblib
from datetime import datetime
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import MWETokenizer
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models.phrases import Phrases

def log(msg):
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print( now + " : " + msg)

class TagPrediction:
    _mPredict = 2
    _mTfidf = 2
    _mMultiLabel = 2
    _mBigram = 2
    
    def __init__(self):
        lDirPath = os.path.dirname(__file__)
        lFilePathToPred = os.path.join(lDirPath, "../model/tags_lr_compressed.joblib")
        lFilePathTfidf= os.path.join(lDirPath, "../model/tfidf.joblib")
        lFilePathMultiLbel = os.path.join(lDirPath, "../model/multilabel.joblib")
        lFilePathToBigram = os.path.join(lDirPath, "../model/bigram.pkl")
        
        if (TagPrediction._mPredict == 2 ) :  
            log("Loading prediction model")                
            TagPrediction._mPredict = joblib.load(lFilePathToPred)
            log("Loading Tfidf")
            TagPrediction._mTfidf =  joblib.load(lFilePathTfidf)
            log("Loading MutiLabel")
            TagPrediction._mMultiLabel =  joblib.load(lFilePathMultiLbel)
            log("Loading bigram")
            TagPrediction._mBigram =  Phrases.load(lFilePathToBigram)
        
        self.init_text_cleaning()
            
        log("TagPrediction object initialized")
        
    def init_text_cleaning(self):
        new_stop_words=['would','want', 'please', 'help', 'can', "can't",'shall','thanks','thank','may',
                'seem','understand','error','warning','require','rather']
        self._mStopWords = stopwords.words('english')
        self._mStopWords.extend(new_stop_words)
        self._mStopWords = set(self._mStopWords)
        self._mLemmatize=WordNetLemmatizer()
        self._mTokenizer = MWETokenizer()
        self._mTokenizer.add_mwe(('c', '#'))
        
    def clean_text(self,text):
        lCleantext = re.sub(r'[^A-Za-z0-9+#.\-]',' ',text.lower())
        lWords = word_tokenize(str(lCleantext.lower()))
        lWords = self._mTokenizer.tokenize(lWords)

        lCleanWords = [str(self._mLemmatize.lemmatize(j)) for j in lWords if j not in self._mStopWords]
        lBigramWords = TagPrediction._mBigram[lCleanWords]

        lCleantext = ' '.join(lBigramWords)
        lCleantext = lCleantext.replace('c_#','c#')

        return lCleantext.strip()
    
    def clean_body(self,body):
        lTxt = BeautifulSoup(body,'html.parser').get_text()
        return self.clean_text(lTxt)
        
    def getPrediction(self,title,body):
        lCleanTxt = self.clean_text(title) + self.clean_body(body)
        lXtfidf = TagPrediction._mTfidf.transform([lCleanTxt])
        lYPred = TagPrediction._mPredict.predict(lXtfidf)
        
        lRes = list(TagPrediction._mMultiLabel.inverse_transform(lYPred)[0])
        
        return lRes     
    
    def getTagList(self):
        return TagPrediction._mMultiLabel.classes_
    
    

if __name__ == "__main__":
    #Unit Testing
    mv = TagPrediction()
    
    print("Test of getTagList ")
    res = mv.getTagList()
    print(res)
    
    print("\n")
    print("Test of getPrediction")
    title = 'Is it possible to use C# Object Initializers with Factories'
    body='<p>I\'m looking at the new object initializers in C# 3.0 and would like to use them. However, I can\'t see how to use them with something like Microsoft Unity. I\'m probably missing something but if I want to keep strongly typed property names then I\'m not sure I can. e.g. I can do this (pseudo code)</p>\n\n<pre><code>Dictionary&lt;string,object&gt; parms = new Dictionary&lt;string,object&gt;();\nparms.Add("Id", "100");\n\nIThing thing = Factory.Create&lt;IThing&gt;(parms)();\n</code></pre>\n\n<p>and then do something in Create via reflection to initialise the parms... but if I want it strongly typed at the Create level, like the new object intitalisers then I don\'t see how I can.</p>\n\n<p>Is there a better way?\nThanks</p>\n'
    res = mv.getPrediction(title,body)
    print(res)
    