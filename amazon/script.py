

import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from nltk import FreqDist
import re, seaborn as sns
import matplotlib.pyplot as plt

# setting required for ease
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



def remove_stop_words(words):
    result = " ".join([word for word in words if word not in stopwords.words('english')])
    return result

def pre_processing():
    dataset = pd.read_csv("./Dataset/train.csv")
    topic_list = dataset.topic.unique()
    dataset = dataset.sort_values(by=['Review Title'], ascending=True).reset_index(drop=True)
    return dataset,topic_list


def analysis(dataset,topic_list):
    '''
        start with some data analysis on Review Text and Review Title
        applying the bag of words approach first
    '''
    # remove stopwords and punctions and symbols
    dataset['Review Text'] = dataset['Review Text'].str.replace("[^a-zA-Z#]", " ")
    # remove short words (length < 3)
    dataset['Review Text'] = dataset['Review Text'].apply(lambda x: ' '.join([w.lower() for w in x.split() if len(w) > 2]))
    all_reviews = [remove_stop_words(words.split(" ")) for words in dataset['Review Text']]

    all_words = ' '.join([word for word in all_reviews]).split()
    '''
        Plotting the top 30 words of highest frequency 
    '''
    freq_dist = FreqDist(all_words)
    words_distribution = pd.DataFrame({'word':list(freq_dist.keys()), 'count':list(freq_dist.values())})
    words_distribution = words_distribution.nlargest(columns='count',n=30) # want to view top 30 words
    plt.figure(figsize=(50,10))
    ax = sns.barplot(data=words_distribution, x="word", y="count")
    ax.set(ylabel='Count')
    plt.show()


if __name__ == "__main__":
    dataset, topic_list = pre_processing()
    analysis(dataset,topic_list)