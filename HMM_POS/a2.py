# coding: utf-8
"""CS585: Assignment 2

In this assignment, you will complete an implementation of
a Hidden Markov Model and use it to fit a part-of-speech tagger.
"""

from collections import Counter, defaultdict
import math
import numpy as np
import os.path
import urllib.request


class HMM:
	def __init__(self, smoothing=0):
		"""
		Construct an HMM model with add-k smoothing.
		Params:
		  smoothing...the add-k smoothing value
		
		This is DONE.
		"""
		self.smoothing = smoothing

	def fit_transition_probas(self, tags):
		"""
		Estimate the HMM state transition probabilities from the provided data.

		Creates a new instance variable called `transition_probas` that is a 
		dict from a string ('state') to a dict from string to float. E.g.
		{'N': {'N': .1, 'V': .7, 'D': 2},
		 'V': {'N': .3, 'V': .5, 'D': 2},
		 ...
		}
		See test_hmm_fit_transition.
		
		Params:
		  tags...a list of lists of strings representing the tags for one sentence.
		Returns:
			None
		"""
		###TODO
		self.transition_probas = defaultdict(lambda : defaultdict(float))
		tag_freqs = defaultdict(lambda : 0)
		
		tag_set = set()
		for sent_tags in tags:
			for idx,tag in enumerate(sent_tags):
				
				tag_set.add(tag)
				
				if idx < len(sent_tags) - 1:
					tag_freqs[tag] += 1
				
				if idx > 0:
					self.transition_probas[sent_tags[idx-1]][sent_tags[idx]] += 1
		
		N = len(tag_set)
		nr = 0
		dr = 0
		
		if self.smoothing > 0:
			nr = 1
			dr = N*self.smoothing

		for tag1 in tag_set:
			for tag2 in tag_set:
				self.transition_probas[tag1][tag2] += nr
				self.transition_probas[tag1][tag2] /= (tag_freqs[tag1] + dr)

				
	def fit_emission_probas(self, sentences, tags):
		"""
		Estimate the HMM emission probabilities from the provided data. 

		Creates a new instance variable called `emission_probas` that is a 
		dict from a string ('state') to a dict from string to float. E.g.
		{'N': {'dog': .1, 'cat': .7, 'mouse': 2},
		 'V': {'run': .3, 'go': .5, 'jump': 2},
		 ...
		}

		Params:
		  sentences...a list of lists of strings, representing the tokens in each sentence.
		  tags........a list of lists of strings, representing the tags for one sentence.
		Returns:
			None		  

		See test_hmm_fit_emission.
		"""
		###TODO
		self.emission_probas = defaultdict(lambda : defaultdict(float))
		tag_freqs = defaultdict(lambda : 0.0)
		tag_set = set()
		word_set = set()
		
		for sent_tag_tup in zip(sentences,tags):
			sentence = sent_tag_tup[0]
			tag_lst  = sent_tag_tup[1]
			
			for word_tag_tup in zip(sentence,tag_lst):
				word = word_tag_tup[0]
				tag = word_tag_tup[1]				
				word_set.add(word)
				tag_set.add(tag)
				tag_freqs[tag] += 1
				
				self.emission_probas[tag][word] += 1
		
		N = len(word_set)
		nr = 0
		dr = 0
		
		if self.smoothing > 0:
			nr = 1
			dr = N*self.smoothing
			
		for word in word_set:
			for tag in tag_set:
				self.emission_probas[tag][word] += nr
				self.emission_probas[tag][word] /= (tag_freqs[tag] + dr)


	def fit_start_probas(self, tags):
		"""
		Estimate the HMM start probabilities form the provided data.

		Creates a new instance variable called `start_probas` that is a 
		dict from string (state) to float indicating the probability of that
		state starting a sentence. E.g.:
		{
			'N': .4,
			'D': .5,
			'V': .1		
		}

		Params:
		  tags...a list of lists of strings representing the tags for one sentence.
		Returns:
			None

		See test_hmm_fit_start
		"""
		###TODO
		self.start_probas = defaultdict(lambda : 0.0)
		tag_set = set()
		
		for tag_lst in tags:
			for idx, tag in enumerate(tag_lst):
				tag_set.add(tag)
				
				if idx == 0:
					self.start_probas[tag] += 1
		
		dr_lhs = len(tags)

		N = len(tag_set)
		nr = 0
		dr = 0
		
		if self.smoothing > 0:
			nr = 1
			dr = N*self.smoothing		
		
		for tag in tag_set:
			self.start_probas[tag] += nr
			self.start_probas[tag] /= (dr_lhs + dr)
			
	def fit(self, sentences, tags):
		"""
		Fit the parameters of this HMM from the provided data.

		Params:
		  sentences...a list of lists of strings, representing the tokens in each sentence.
		  tags........a list of lists of strings, representing the tags for one sentence.
		Returns:
			None		  

		DONE. This just calls the three fit_ methods above.
		"""		
		self.fit_transition_probas(tags)
		self.fit_emission_probas(sentences, tags)
		self.fit_start_probas(tags)


	def viterbi(self, sentence):
		"""
		Perform Viterbi search to identify the most probable set of hidden states for
		the provided input sentence.

		Params:
		  sentence...a lists of strings, representing the tokens in a single sentence.

		Returns:
		  path....a list of strings indicating the most probable path of POS tags for
		  		  this sentence.
		  proba...a float indicating the probability of this path.
		"""
		###TODO
		pass


def read_labeled_data(filename):
	"""
	Read in the training data, consisting of sentences and their POS tags.

	Each line has the format:
	<token> <tag>

	New sentences are indicated by a newline. E.g. two sentences may look like this:
	<token1> <tag1>
	<token2> <tag2>

	<token1> <tag1>
	<token2> <tag2>
	...

	See data.txt for example data.

	Params:
	  filename...a string storing the path to the labeled data file.
	Returns:
	  sentences...a list of lists of strings, representing the tokens in each sentence.
	  tags........a lists of lists of strings, representing the POS tags for each sentence.
	"""
	###TODO
	pass

def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/ty7cclxiob3ajog/data.txt?dl=1'
    urllib.request.urlretrieve(url, 'data.txt')

if __name__ == '__main__':
	"""
	Read the labeled data, fit an HMM, and predict the POS tags for the sentence
	'Look at what happened'

	DONE - please do not modify this method.

	The expected output is below. (Note that the probability may differ slightly due
	to different computing environments.)

	$ python3 a2.py  
	model has 34 states
        ['$', "''", ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', '``']
	predicted parts of speech for the sentence ['Look', 'at', 'what', 'happened']
	(['VB', 'IN', 'WP', 'VBD'], 2.751820088075314e-10)
	"""
	fname = 'data.txt'
	if not os.path.isfile(fname):
		download_data()
	sentences, tags = read_labeled_data(fname)

	model = HMM(.001)
	model.fit(sentences, tags)
	print('model has %d states' % len(model.states))
	print(model.states)
	sentence = ['Look', 'at', 'what', 'happened']
	print('predicted parts of speech for the sentence %s' % str(sentence))
	print(model.viterbi(sentence))
