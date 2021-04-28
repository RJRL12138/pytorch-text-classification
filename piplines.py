import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2


nltk.download('punkt')
training = []
stop_words = stopwords.words('english')
stop_words.append('N')
stop_words = set(stop_words)
porter = PorterStemmer()
with open('training.csv','r',encoding='utf-8') as f:
    fr = csv.reader(f)
    header = next(fr)
    for line in fr:
        text = line[0]
        tag = line[1]
        tokens = word_tokenize(text.lower())
        words = [porter.stem(w) for w in tokens if w not in stop_words]
        training.append([words, tag])
with open('train_token.csv','w',encoding='utf-8') as w:
    w.write("text,label\n")
    for item in training:
        w.write('+'.join(item[0])+","+item[1]+'\n')
