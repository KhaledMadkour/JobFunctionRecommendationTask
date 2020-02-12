from flask import Flask, jsonify, request
from sklearn.naive_bayes import GaussianNB
import pickle
import numpy as np
from skmultilearn.problem_transform import BinaryRelevance
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import ast
import re

nltk.download('stopwords')

app = Flask(__name__) #define app using Flask


model = pickle.load(open("models/br_classifier", 'rb'))
vectorizer = pickle.load(open("models/TfidfVectorizer.pk", 'rb'))


def classes():
	df = pd.read_csv("jobs_data.csv")
	df['jobFunction'] = df['jobFunction'].apply(ast.literal_eval).apply(np.sort)
	unique_jobfn = []
	for i in df.jobFunction:
	  for x in i:
	  	if(x !='nan'):
	  		unique_jobfn.append(x)
	return(list(set(unique_jobfn)))

@app.route('/<string:title>', methods=['GET'])
def test(title):
	cl = classes()
	cl.sort()
	stemmer = PorterStemmer()
	words = stopwords.words("english")
	usr = " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", title).split() if i not in words]).lower()
	uq_vectorized = vectorizer.transform(np.array([usr]))
	usr_predictions  = model.predict(uq_vectorized)

	ll = []
	for t in (usr_predictions):
	  for i in t.indices:
	    ll.append(cl[i])

	return jsonify(ll)


if __name__ == '__main__':
	app.run(debug=True , port = 8080) #run app on port 8080 in debug mode