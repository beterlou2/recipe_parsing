import tensorflow as tf
from tensorflow.compat.v1 import global_variables_initializer as global_variables_initializer
from tensorflow.compat.v1 import tables_initializer as tables_initializer
from tensorflow.compat.v1 import Session as Session

import tensorflow_hub as hub
import pandas as pd
import numpy as np
import spacy
from scipy import spatial

from flair.models import SequenceTagger
from flair.data import Sentence
from segtok.segmenter import split_single

class action_classifier:
	def __init__(self):
		self.nlp = spacy.load('en_core_web_sm')
		url = "https://tfhub.dev/google/elmo/2"
		self.embed = hub.Module(url)

		self.tagger = SequenceTagger.load('pos')
		self.verb_classes = []
		self.class_length = None
		self.action_vec = []

	def split_str(self, string):
		temp = string[1:len(string)-1].split(',')
		result = []
		for word in temp:
			word = word.strip()
			word = word[1:len(word)-1]
			result.append(word)

		return result

	def create_dict(self):
		
		self.verb_classes = np.asarray(pd.read_csv('EPIC_verb_classes')['class_key'])
		self.class_length = self.verb_classes.shape[0]

		verb_classes_total = np.asarray(pd.read_csv('EPIC_verb_classes')['verbs'])

		idx = 0
		all_vec = []
		lookup = []
		for i in range(verb_classes_total.shape[0]):
			temp = self.split_str(verb_classes_total[i])
			for j in temp:
				all_vec.append(j)
				idx += 1
			lookup.append(idx)

		action_embeddings = self.embed(all_vec, signature="default", as_dict=True)["default"]

		action_vector = []
		with Session() as sess:
			sess.run(global_variables_initializer())
			sess.run(tables_initializer())
			action_vector = sess.run(action_embeddings)

		for i in range(len(lookup)):
			if i == 0:
				temp_vec = np.asarray(action_vector[0:lookup[i], :])
				avg_vec = np.mean(temp_vec)
				self.action_vec.append((self.verb_classes[i], avg_vec))
			else:
				temp_vec = np.asarray(action_vector[lookup[i-1]:lookup[i], :])
				avg_vec = np.mean(temp_vec)
				self.action_vec.append((self.verb_classes[i], avg_vec))

	def classify(self, embeddings):
		verb_classes = []
		for ebd in embeddings:
			max_sim = 0
			verb_class = None
			for i in range(self.class_length):
				sim = 1 - spatial.distance.cosine(ebd, self.action_vec[i][1])
				
				if sim > max_sim:
					max_sim = sim
					verb_class = self.action_vec[i][0]
			verb_classes.append(verb_class)

		return verb_classes

	def pos_tag(self, text):
		sentences = [Sentence(sent, use_tokenizer=True) for sent in split_single(text)]
		sent2act = []
		sent_len = []

		for sent in sentences:
			self.tagger.predict(sent)
			count = 0
			actions = []
			for token in sent:
				tag = token.get_tag(tag_type='pos').value

				if tag[0:2] == 'VB':
					actions.append(token.text)
					count +=  1
			sent2act += actions
			sent_len.append(count)

		return sent2act, sent_len

	def parse_file(self, filename):
		file = open(filename, 'r')
		text = file.read()

		text = text.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')
		text = " ".join(text.split())

		sentences, sent_lookup = self.pos_tag(text)

		embeddings = self.embed(sentences, signature="default", as_dict=True)["default"]

		x = []
		with Session() as sess:
			sess.run(global_variables_initializer())
			sess.run(tables_initializer())
			x = sess.run(embeddings)
		
		verb_classes = self.classify(x)

		result = []
		aux = []
		idx = 0
		for i in sent_lookup:
			result.append(verb_classes[idx:idx+i])
			aux.append(sentences[idx:idx+i])
			idx += i
		
		return aux, result


ac = action_classifier()
ac.create_dict()
aux, sent = ac.parse_file('recipe.txt')
for i in range(len(aux)):
	print(aux[i])
	print(sent[i])
	print()





