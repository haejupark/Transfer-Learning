import re
import numpy as np
from settings import *
from keras.utils import *
from keras.preprocessing import sequence

def tokenize(data):
	return [x.strip().lower() for x in re.split('(\W+)', data) if x.strip()]

def word_dict(data):
	word_dict = {}
	word_dict['ㄱ'] = len(word_dict)
	word_dict['UNK'] = len(word_dict)
	
	for line in data:
		q1 = tokenize(line[1])
		q2 = tokenize(line[2])
		for word in q1:
			if word not in word_dict:
				word_dict[word] = len(word_dict)
		for word in q2:
			if word not in word_dict:
				word_dict[word] = len(word_dict)
				
	return word_dict

def get_vectors(word_dict, vec_file, emb_size):
	word_vectors = np.random.uniform(-0.1, 0.1, (len(word_dict), emb_size))
	
	f = open(vec_file, 'r', encoding='utf-8')
	vec = {}
	for line in f:
		line = line.split()
		vec[line[0]] = np.array([float(x) for x in line[-emb_size:]])
	f.close()
    
	for key in word_dict:
		low = key.lower()
		if low in vec:
			word_vectors[word_dict[key]] = vec[low]

	return word_vectors

def map_to_id(data, vocab):
	return [vocab[word] if word in vocab else 1 for word in data]
	
def load_data(data, word_dict, labels=labels):
	X,Y,Z = [], [], []
	for label, q1, q2 in data:
		q1 = map_to_id(tokenize(q1), word_dict)
		q2 = map_to_id(tokenize(q2), word_dict)
		
		if len(q1) > max_len_sentence:
			q1 = q1[:max_len_sentence]
		if len(q2) > max_len_sentence:
			q2 = q2[:max_len_sentence]
		
		if label in labels:
			X+= [q1]
			Y+= [q2]
			Z+= [labels[label]]
	X = sequence.pad_sequences(X, maxlen = max_len_sentence)
	Y = sequence.pad_sequences(Y, maxlen = max_len_sentence)
	Z = to_categorical(Z)
	return X, Y, Z
