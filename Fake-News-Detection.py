import itertools
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import tokenize
from sklearn import metrics, model_selection, naive_bayes, pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from wordcloud import WordCloud

true = pd.read_csv("data/True.csv")
fake = pd.read_csv("data/Fake.csv")

lst_stopwords = nltk.corpus.stopwords.words("english")
token_space = tokenize.WhitespaceTokenizer()

true['target'] = 'true'
fake['target'] = 'fake'

data = pd.concat([fake, true]).reset_index(drop=True)

data = shuffle(data).reset_index(drop=True)

data.drop(["date", "title"], axis=1, inplace=True)


def process_text(text, lst_stopwords=None):
    ## convert to lower case
    text = text.lower()

    ##Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    ## Tokenize (convert from string to list)
    lst_text = text.split()

    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    ## back to string from list
    text = " ".join(lst_text)

    return text


def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({
        "Word": list(frequency.keys()),
        "Frequency": list(frequency.values())
    })
    df_frequency = df_frequency.nlargest(columns="Frequency", n=quantity)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_frequency, x="Word", y="Frequency", color='blue')
    ax.set(ylabel="Count")
    plt.xticks(rotation='vertical')
    plt.show()


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


data['text'] = data.text.apply(lambda x: process_text(x, lst_stopwords))

# # print(data)
print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
# plt.show()

print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind='bar')
# plt.show()

fake_data = data[data["target"] == "fake"]
all_words1 = ' '.join([text for text in fake_data.text])
wordcloud1 = WordCloud(width=800,
                       height=500,
                       max_font_size=110,
                       collocations=False).generate(all_words1)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis("off")
# plt.show()

true_data = data[data["target"] == "true"]
all_words2 = ''.join([text for text in true_data.text])
wordcloud2 = WordCloud(width=800,
                       height=500,
                       max_font_size=110,
                       collocations=False).generate(all_words2)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
# plt.show()

## Most frequent words in fake news
# counter(data[data["target"] == "fake"], "text", 20)
# ## Most frequent words in real news
# counter(data[data["target"] == "true"], "text", 20)

##Split data
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    data['text'], data.target, test_size=0.2, random_state=42)

##Naive bayes

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

# Vectorizing and applying TF-IDF
pipe = pipeline.Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('model', classifier)])
# Fitting the model
model = pipe.fit(X_train, y_train)
# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(
    accuracy_score(y_test, prediction) * 100, 2)))
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
plt.show()

# ##Decision Tree

# from sklearn import tree
# classifier = tree.DecisionTreeClassifier()

# # Vectorizing and applying TF-IDF
# pipe = pipeline.Pipeline([('vect', CountVectorizer()),
#                  ('tfidf', TfidfTransformer()),
#                  ('model', classifier)])
# # Fitting the model
# model = pipe.fit(X_train, y_train)
# # Accuracy
# prediction = model.predict(X_test)
# print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
# cm = metrics.confusion_matrix(y_test, prediction)
# plot_confusion_matrix(cm, classes=['Fake', 'Real'])

# ##Randomforest

# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(max_depth=2, random_state=0)

# # Vectorizing and applying TF-IDF
# pipe = pipeline.Pipeline([('vect', CountVectorizer()),
#                  ('tfidf', TfidfTransformer()),
#                  ('model', classifier)])
# # Fitting the model
# model = pipe.fit(X_train, y_train)
# # Accuracy
# prediction = model.predict(X_test)
# print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
# cm = metrics.confusion_matrix(y_test, prediction)
# plot_confusion_matrix(cm, classes=['Fake', 'Real'])
