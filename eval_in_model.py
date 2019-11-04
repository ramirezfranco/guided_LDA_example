import warnings
import pandas as pd
warnings.filterwarnings(action='ignore', category=UserWarning)
import gensim
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim.test.utils import datapath
import multiprocessing as mp
import sys
import json
from flask import render_template
import flask
import os


'''
1. The following lines contain different objects used in the utility functions
'''
stemmer = SnowballStemmer("english")
sw = list(stopwords.words('english'))
best_model_path = os.path.abspath('example_lda_model')
best_dictionary_path = os.path.abspath('best_dictionary')
model = gensim.models.ldamodel.LdaModel.load(datapath(best_model_path))
dictionary = gensim.corpora.Dictionary.load(datapath(best_dictionary_path))


'''
2. Obtaining parameters from the command line
'''
if len(sys.argv) > 1:
	file_name = sys.argv[1]
	interest_topic = sys.argv[2]
	outfile_name = file_name[:-4]+'_'+interest_topic
else:
	print('you need to provide a csv file name and a topic of interest')


'''
3. Reading the collection of documents
'''
# Uncomment the following lines when colum names are not provided in the csv file
#df = pd.read_csv(file_name, header=None)
#df.columns = ['id', 'sec_id', 'text']
df = pd.read_csv(file_name)


'''
4. Storing some parameters used in the process. This information will be 
transmited to the html render.
'''
params = {
	'model':best_model_path,
	'csv_input': file_name,
	'interest_topic': interest_topic,
	'topic_terms': ', '.join([dictionary[w] for w,f in model.get_topic_terms(int(interest_topic), 10)]),
	'total': len(df)
}


'''
5. Utility functions
'''
def csv2corpus(df, id_column, text_colum):
	'''
	Takes a data frame with the corpus information and convert it to a 
	list of tuples with id and text, that is used as input of the multipro-
	cessing function "map_async" method.
	Inputs: 
		- df (pandas data fram): 
		- id_column (string):
		- text_column (string):

	'''
	corpus = [(r[id_column],r[text_colum]) for i, r in df.iterrows()]
	return corpus


def doc2vec(text):
	'''
	Coverts a raw text into a vector representation, based on the 
	best_dictionary.
	Inputs:
		- text(string): raw text.
	'''
	corp = nltk.word_tokenize(text)
	corp = [t.lower() for t in corp if t not in sw]
	corp = [t for t in corp if len(t) > 2]
	corp = [stemmer.stem(t) for t in corp]

	doc_vec = dictionary.doc2bow(corp)

	return doc_vec


def eval_document_vector(doc_tup, interest_topic=interest_topic, th=0.85):
	'''
	Takes one id-text tupple and identifies if the text contains the topic
	of interest and if the probability of that topic in the text is greater
	than a threshold.
	Inputs:
		- doc_tup (tupple): Tupple that contain a unique id in the first position
		  and a cleaned text in the second one.
		- interest_topic (string): Number of the topic that we are interested on. 
		- th (float): minimum probability that a topic must have to be consider 
		  as present in the text.
	'''
	doc_vec = doc2vec(doc_tup[1])
	topics = model[doc_vec][0]
	content = {t[0]:t[1] for t in topics}
	print('Processing document {}'.format(doc_tup[0]))
	if int(interest_topic) in content.keys():
		if content[int(interest_topic)] > th:
			return doc_tup[0]


'''
6. Creating the input used in the multiprocessing Pool
'''
corpus = csv2corpus(df, df.columns[0], df.columns[2])


'''
7. Creating the flask app tha is used to create the HTML file with the results
'''
app = flask.Flask('my app')


'''
8. Multiprocessing part
'''
if __name__ == '__main__':
    p = mp.Pool(processes=3)
    r = p.map_async(eval_document_vector, corpus)

    p.close()
    p.join()
    
    results = r.get()
    results = [r for r in results if r]

    df.set_index(df.columns[0], inplace=True)
    df = df.loc[results]

    final_results = {i:{df.columns[j]:r[df.columns[j]] for j in range(len(df.columns))} for i, r in df.iterrows()}
# Uncomment the following two lines if you want to store the results as a json file
    # with open(outfile_name+'.json', 'w') as outfile:
    # 	json.dump(final_results, outfile)

    with app.app_context():
        rendered = render_template(
        	'eval_topics.html',
        	vars= params, 
        	columns=df.columns, 
        	data=final_results,
        	docs_found=len(final_results.keys()))

        with open(outfile_name+'.html', "w") as file:
            file.write(rendered)

        print('HTML file with results created')