from sklearn.model_selection import train_test_split
import pandas as pd
import regex as re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np
nltk.download('stopwords')

# Data import

ds = pd.read_csv('onion-or-not.csv')
                 #, nrows=12001)


def titles_to_words(raw_title):

    # Punctuation removal
    letters_only = re.sub("[^a-zA-Z]", " ", raw_title)

    # no Upper letters
    words = letters_only.lower().split()

    # Word stemming for every word
    stemmer = PorterStemmer()
    tokens_lem = [stemmer.stem(i) for i in words]

    # Stop words removal
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]

    # Join into sentence again
    return (" ".join(meaningful_words))



# total_titles = ds.shape[0]

# Titles after preprocessing
clean_titles = []

j = 0
for title in ds['text']:
    # Convert to words, then append to clean_train
    clean_titles.append(titles_to_words(title))

# Create Tf_Idf Bag of Words
vect = TfidfVectorizer()

# Fit the vectorizer on our corpus and transform
matrix_titles = vect.fit_transform(clean_titles)
matrix_titles = pd.DataFrame(matrix_titles.toarray(), columns=vect.get_feature_names())
matrix_labels = ds['label']

# Create Classifier Neural Network, Train it and run predictions
titles_train, titles_test, label_train, label_test = train_test_split(matrix_titles, matrix_labels, test_size=0.25, random_state=15)

classifier = MLPClassifier()
classifier.fit(titles_train, label_train)

predictions = classifier.predict(titles_test)
print(classification_report(label_test, predictions))
