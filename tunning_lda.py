import warnings
import pandas as pd
warnings.filterwarnings(action='ignore', category=UserWarning)
import gensim
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from gensim.models import TfidfModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem.snowball import SnowballStemmer
import re
from gensim.test.utils import datapath
regex_tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")
wn = WordNetLemmatizer()



def create_eta(priors, etadict, ntopics):
    '''
    Creates an eta matrix to specify the important terms that a topic most contain.
    Inputs:
        - priors (dict): dictionary where every key is a term (str) and every value 
          is the number of topic (int) where the term most appear.
        - etadic (dict): dictionary produced by the training corpus.
        - ntopics (int): number of topics in the model.
    '''
    eta = np.full(shape=(ntopics, len(etadict)), fill_value=1) # create a (ntopics, nterms) matrix and fill with 1
    for word, topic in priors.items(): # for each word in the list of priors
        keyindex = [index for index,term in etadict.items() if term==word] # look up the word in the dictionary
        if (len(keyindex)>0): # if it's in the dictionary
            eta[topic,keyindex[0]] = 1e7  # put a large number in there
    eta = np.divide(eta, eta.sum(axis=0)) # normalize so that the probabilities sum to 1 over all topics
    return eta

def make_LDA(vec_corpus, dictionary, n, prior='auto', iter_v=200, pass_v=150, decay_v=0.7):
    '''
    Creates an LDA model with the given parameters.
    Inputs:
        - vec_corpus: Stream of document vectors or sparse matrix.
        - dictionary (dict): Mapping from word IDs to words.
        - n (int): The number of requested latent topics to be extracted from the 
          training corpus.
        - prior (dict): Map from terms to topics.
        - iter_v (int): number os maximum iterations.
        - pass_v (int): number of passes through the corpus.
        - deccay_v (float): A number between (0.5, 1].
    '''
    if prior =='auto':
        eta_matrix = 'auto'
    else:
        eta_matrix = create_eta(prior, dictionary, n)
    model = gensim.models.ldamodel.LdaModel(
        corpus=vec_corpus, 
        id2word=dictionary, 
        num_topics=n,
        random_state=42, 
        eta=eta_matrix,
        iterations=iter_v,
        eval_every=-1, 
        update_every=1,
        passes=pass_v, 
        alpha='auto', 
        per_word_topics=True,
        decay = decay_v
    )
    return model

def evaluate_model(model, corpus, dictionary, n):
    '''
    Computes the perplexity, coherence and topics of an LDA model.
    Inputs:
        - model: An LDA model produced using Gensim.
        - corpus: The corpus used to train the model.
        - dictionary: the dictionary created by the corpus.
        - n: number of topics of the model
    '''
    perplexity = model.log_perplexity(corpus)
    topics = [[dictionary[w] for w,f in model.get_topic_terms(topic, 10)] for topic in range(n)]
    cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence = cm.get_coherence()
    return topics, perplexity, coherence

def dif_models(vec_corpus, dictionary, n_list, iter_list, pass_list, decay_list, prior="auto"):
    '''
    Computes different LDA models and preserve the one with the best coherence.
    Inputs:
        - vec_corpus: the corpus used to train the model.
        - dictionary: the dictionary created by the corpus
        - n_list: list of different numbers of topics.
        - iter_list: list of different numbers of iterations values.
        - pass_list: list of different pass values.
        - decay_list: list of different decay values.
        - prior (dict): dictionary where every key is a term (str) and every value 
          is the number of topic (int) where the term most appear.
    '''
    best_model = None
    best_coherence = -10000
    best_perplexity = -10000
    topics_bm = None
    results = []
    for n in n_list:
        for i in iter_list:
            for p in pass_list:
                for d in decay_list:
                    lda = make_LDA(vec_corpus, dictionary, n, prior, i, p, d)
                    topics, perp, cohe = evaluate_model(lda, vec_corpus, dictionary, n)
                    results.append([n, i, p, d, perp, cohe])
                    if cohe > best_coherence:
                        best_model = lda
                        topics_bm = topics
                        best_coherence = cohe
                        best_perplexity = perp
    colum_names = ['Number of Topics', 'Iterations', 'Passes', 'Decay', 'Perplexity', 'Coherence']
    df = pd.DataFrame(results, columns=colum_names)
    df = df.sort_values(by='Coherence', ascending=False)
    return df, best_model, topics_bm

def print_topics(topics):
    '''
    Print the top 10 most important terms of every topic of an LDA model.
    Inputs:
        - topics (list): list of lists of terms by topic, produced by an LDA model.
    '''
    for topic in range(len(topics)):
        print('Topic ', topic)
        print(', '.join(topics[topic]))
        print('**************************************************************************************************')
        
def save_model(obj, name):
    f = datapath(name)
    obj.save(f)
    