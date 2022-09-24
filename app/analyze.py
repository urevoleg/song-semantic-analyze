import os
import re
import datetime as dt
from pprint import pprint

from typing import List, DefaultDict

from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from app import logger, db
from app.models import Text, Song
from app.sources import SourceSong, MostFreqArtist

from sqlalchemy import or_

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import string


corpus = ['британская полиция знает о местонахождении основателя wikileaks',
            'в суде сша начинается процесс против россиянина, рассылавшего спам',
            'церемонию вручения нобелевской премии мира бойкотируют 19 стран',
            'в великобритании арестован основатель сайта wikileaks джулиан ассандж',
            'украина игнорирует церемонию вручения нобелевской премии',
            'шведский суд отказался рассматривать апелляцию основателя wikileaks',
            'нато и сша разработали планы обороны стран балтии против россии',
            'полиция великобритании нашла основателя wikileaks, но, не арестовала',
            'в стокгольме и осло сегодня состоится вручение нобелевских премий']


def get_text(corpus: List[str]=corpus) -> str:
    for text in corpus:
        yield text


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language: str='russian'):
        self.language = language
        self.stopwords = set(stopwords.words(language))
        self.lemmantizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer(language)

    def is_stopworld(self, token: str) -> bool:
        return token in self.stopwords

    def is_punct(self, token: str) -> bool:
        return token in string.punctuation

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        for token in word_tokenize(text):
            yield token
            #yield self.stemmer.stem(token)

    def lemmantize(self, token):
        return self.lemmantizer.lemmatize(token)

    def normalize(self, text):
        return [
            self.lemmantize(token)
            for token in self.tokenize(text)
            if not self.is_punct(token) and not self.is_stopworld(token)
        ]

    def join_normalize(self, text):
        return " ".join(self.normalize(text))

    def fit(self, X, y=None):
        return self

    def transform(self, corpus):
        for text in corpus:
            yield self.join_normalize(text)


class TopicModels():

    def __init__(self, n_topics=2, estimator='LSA', language='english', vect='count'):
        self.n_topics = n_topics
        self.language = language
        if estimator == 'LSA':
            self.estimator = TruncatedSVD(n_components=self.n_topics, random_state=88)
        if estimator == 'LDA':
            self.estimator = LatentDirichletAllocation(n_components=self.n_topics, random_state=88,
                                                        learning_method="online",
                                                        learning_offset=50.0)
        if estimator == 'NMF':
            self.estimator = NMF(n_components=self.n_topics, random_state=88)

        if vect == 'count':
            self.vectorizer = CountVectorizer(preprocessor=None, lowercase=True,
                                              max_df=0.25, min_df=3, max_features=25,
                                              stop_words='english')
        if vect == 'tfidf':
            self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 3),
                                              max_df=0.25, min_df=3, max_features=25,
                                              stop_words='english')

        self.model = Pipeline([
            #("norm", TextNormalizer(language=self.language)),
            ("vect", self.vectorizer),
            ("model", self.estimator),
            ("reduction", TSNE(n_components=2, n_jobs=os.cpu_count()//2, random_state=88))
        ], verbose=1)

    def fit_transform(self, corpus):
        self.trancated_corpus = self.model.fit_transform(corpus)
        return self.model

    def get_topics(self, n_lexemes=5):
        vectorizer = self.model.named_steps['vect']
        model = self.model.named_steps['model']
        names = vectorizer.get_feature_names_out()
        topics = {}

        for idx, topic in enumerate(model.components_):
            features = topic.argsort()[:-(n_lexemes-1):-1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens
        return topics


class SongCorpusReader():
    def __init__(self,
                 list_of_song_id=set([1805]),
                 list_of_artist_id=set([12968, 55403, 48537, 9941, 39218])):
        """
        :param list_of_song_id:
        :param list_of_artist_id: (12968 - Diana Krall, 55403 - Within Temptation, 48537 - The Beatles)
        """
        self.list_of_song_id = list_of_song_id
        self.list_of_artist_id = list_of_artist_id

    def get_text(self):
        q = db.session.query(Text.text)\
                .outerjoin(Song, Text.song_id == Song.song_id)\
                .filter(or_(Song.song_id.in_(self.list_of_song_id),
                            Song.artist_id.in_(self.list_of_artist_id)))
        for row in q:
            yield row.text


def plot_top_words(model, feature_names, n_top_words, title, n_topics):
    fig, axes = plt.subplots(ncols=n_topics, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


if __name__ == '__main__':
    logger.info(dt.datetime.now())
    n_topics = 3
    most_freq_artists = db.session.query(MostFreqArtist.artist_id)
    corpus = SongCorpusReader(list_of_artist_id=[row.artist_id for row in most_freq_artists]).get_text()

    lsa = TopicModels(n_topics=n_topics, estimator='LSA', language='english', vect='tfidf')
    lsa.fit_transform(corpus)

    print(lsa.model.named_steps['model'].components_.shape, lsa.trancated_corpus.shape)

    topics = lsa.get_topics(n_lexemes=10)

    for topic, terms in topics.items():
        print(f"Topic: {topic}")
        print(terms)

    logger.info(dt.datetime.now())

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.despine(ax=ax)
    ax.plot(lsa.trancated_corpus[:, 0], lsa.trancated_corpus[:, 1], marker='s', ms=5, ls='none', alpha=0.25)
    ax.grid(axis="both", lw=0.25, color='grey')
    plt.show()

    tf_feature_names = lsa.model.named_steps['vect'].get_feature_names_out()
    plot_top_words(lsa.model.named_steps['model'], tf_feature_names, 10, "Topics in LSA model", n_topics)










