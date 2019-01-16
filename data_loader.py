import re
import numpy as np
from settings import *
from keras.utils import *
from keras.preprocessing import sequence
from keras.layers import *
import settings

setting = settings.Setting()
def tokenize(data):
	return [x.strip().lower() for x in re.split('(\W+)', data) if x.strip()]

def get_vocab(data):
	word_dict = {}
	word_dict['PAD'] = len(word_dict)
	word_dict['UNK'] = len(word_dict)
	for line in data:
		#print(data)
		q1 = tokenize(line[1])
		try:
			q2 = tokenize(line[2])
		except:
			print(line)

		for word in q1:
			if word not in word_dict:
				word_dict[word] = len(word_dict)
		for word in q2:
			if word not in word_dict:
				word_dict[word] = len(word_dict)				
	return word_dict

def get_embeddings(word_dict, vec_file, emb_size):
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
	
def load_data(data, word_dict, task="snli", labels=labels):
	X,Y,Z = [], [], []
	for label, q1, q2 in data:
		q1 = map_to_id(tokenize(q1), word_dict)
		q2 = map_to_id(tokenize(q2), word_dict)	
		if len(q1) > setting.max_len:
			q1 = q1[:setting.max_len]
		if len(q2) > setting.max_len:
			q2 = q2[:setting.max_len]
		if task == "snli":
			X+= [q1]
			Y+= [q2]
			Z+= [labels[label]]		
		else:
			X+= [q1]
			Y+= [q2]
			Z+= [label]
	X = sequence.pad_sequences(X, maxlen = setting.max_len)
	Y = sequence.pad_sequences(Y, maxlen = setting.max_len)
	if task =="snli":
		Z = to_categorical(Z,num_classes=3)
	else:
		Z = to_categorical(Z,num_classes=2)
	return X, Y, Z

def create_train_dev_set(train_data, dev_data, word_dict, labels=labels):
	train_X,train_Y,train_Z = [], [], []
	for label, q1, q2 in train_data:
		q1 = map_to_id(tokenize(q1), word_dict)
		q2 = map_to_id(tokenize(q2), word_dict)	
		if len(q1) > setting.max_len:
			q1 = q1[:setting.max_len]
		if len(q2) > setting.max_len:
			q2 = q2[:setting.max_len]
		train_X+= [q1]
		train_Y+= [q2]
		train_Z+= [labels[label]]		

	train_X = sequence.pad_sequences(train_X, maxlen = setting.max_len)
	train_Y = sequence.pad_sequences(train_Y, maxlen = setting.max_len)
	train_Z = to_categorical(train_Z,num_classes=3)

	dev_X,dev_Y,dev_Z = [], [], []
	for label, q1, q2 in dev_data:
		q1 = map_to_id(tokenize(q1), word_dict)
		q2 = map_to_id(tokenize(q2), word_dict)	
		if len(q1) > setting.max_len:
			q1 = q1[:setting.max_len]
		if len(q2) > setting.max_len:
			q2 = q2[:setting.max_len]
		dev_X+= [q1]
		dev_Y+= [q2]
		dev_Z+= [labels[label]]		

	dev_X = sequence.pad_sequences(dev_X, maxlen = setting.max_len)
	dev_Y = sequence.pad_sequences(dev_Y, maxlen = setting.max_len)
	dev_Z = to_categorical(dev_Z,num_classes=3)
	
	return train_X, train_Y, train_Z, dev_X, dev_Y, dev_Z
	

	
def submult(input_1, input_2):
    mult = Multiply()([input_1, input_2])
    sub = Lambda(lambda x: K.abs(x[0] - x[1]))([input_1,input_2])
    out_= Concatenate()([sub, mult])
    return out_
	

def acc(y_true, y_pred):
    return np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)).mean()

	
