import gensim.downloader as api
import nltk
from gensim.corpora import Dictionary
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
docs = api.load('text8')

dictionary = Dictionary()
for doc in docs:
    dictionary.add_documents([[lemmatizer.lemmatize(token) for token in doc]])
dictionary.filter_extremes(no_below=20, no_above=0.5)

corpus = [dictionary.doc2bow(doc) for doc in docs]