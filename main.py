import numpy as np
import pandas as pd
import argparse
import settings
import os
import time
from settings import *
from data_loader import *
from keras.layers import *
from keras.activations import softmax
from keras.regularizers import l2
from keras.models import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from flipGradientTF import *
from keras.utils import *

np.random.seed(3)

class Multimodel(object):
	def __init__(self,setting,is_train=True,is_adv=True):
		self.lr = setting.lr
		self.word_dim = setting.word_dim
		self.lstm_dim = setting.lstm_dim
		self.max_len = setting.max_len
		self.dense_dim = setting.dense_dim
		self.keep_prob = setting.keep_prob
		self.batch_size = setting.batch_size
		self.drop_prob = 1 - self.keep_prob
		self.optimizer = Adam(self.lr)
		self.epochs = setting.epochs
		self.is_adv = is_adv
		self.is_train = is_train
		self.lambda1 = setting.lambda1
		
		self.train_dir = setting.train_dir
		self.dev_dir = setting.dev_dir
		self.embed_dir = setting.embed_dir

	def adversarial_loss(self, prem, hypo, label):		
		# Gradient Reversal Layer
		flip_gradient = GradientReversal(1)
		flip_prem = flip_gradient(prem)
		flip_hypo = flip_gradient(hypo)		
		# merge and pass it to dense layer	
		merged = Concatenate()([flip_prem, flip_hypo, submult(flip_prem,flip_hypo)])
		dense = Dropout(self.drop_prob)(merged)
		dense = Dense(self.dense_dim, kernel_initializer='uniform', activation='relu')(dense)
		dense = Dropout(self.drop_prob)(dense)
		out = Dense(2, activation='softmax')(dense)
		adv_loss = K.mean(K.categorical_crossentropy(label, out))
		return adv_loss
		
	def classifier(self, features):
		dense = Dropout(self.drop_prob)(features)
		dense = Dense(self.dense_dim, kernel_initializer='uniform', activation='relu')(dense)
		dense = Dropout(self.drop_prob)(dense)
		preds = Dense(3, activation='softmax')(dense)
		return preds
		
	def train_model(self, source_language, target_language):	
		# Load source data (english)
		source_train_name = self.train_dir+"multinli.train.%s.txt"%source_language
		source_dev_name = self.dev_dir+"xnli_%s.txt"%source_language
		source_embeddings_name = self.embed_dir+"wiki.%s.vec"%source_language
		
		source_train_data=[l.strip().split('\t') for l in open(source_train_name, errors='ignore')]
		source_dev_data=[l.strip().split('\t') for l in open(source_dev_name, errors='ignore')]

		source_word_vocab = get_vocab(source_train_data + source_dev_data)	
		source_word_embeddings = get_embeddings(source_word_vocab, source_embeddings_name, setting.word_dim)

		source_train_X, source_train_Y, source_train_Z, \
			source_val_X, source_val_Y, source_val_Z = create_train_dev_set(source_train_data, source_dev_data, source_word_vocab)
		
		if source_train_X is None:
			print("++++++ Unable to train model +++++++")
			return None	
		############################################################
		# Load target data (translated other language)
		target_train_name = self.train_dir+"multinli.train.%s.txt"%target_language
		target_dev_name = self.dev_dir+"xnli_%s.txt"%target_language
		target_embeddings_name = self.embed_dir+"wiki.%s.vec"%target_language
		
		target_train_data=[l.strip().split('\t') for l in open(target_train_name, errors='ignore')]
		target_dev_data=[l.strip().split('\t') for l in open(target_dev_name, errors='ignore')]

		target_word_vocab = get_vocab(target_train_data + target_dev_data)	
		target_word_embeddings = get_embeddings(target_word_vocab, target_embeddings_name, setting.word_dim)

		target_train_X, target_train_Y, target_train_Z, \
			target_val_X, target_val_Y, target_val_Z = create_train_dev_set(target_train_data, target_dev_data, target_word_vocab)
		
		if target_train_X is None:
			print("++++++ Unable to train model +++++++")
			return None	
		#############################################################		
		# Word embedding layer for source and target language	
		source_embedding_layer = Embedding(len(source_word_vocab), self.word_dim, weights=[source_word_embeddings], input_length=(self.max_len,), trainable=False)
		target_embedding_layer = Embedding(len(target_word_vocab), self.word_dim, weights=[target_word_embeddings], input_length=(self.max_len,), trainable=False)
		
		# Input
		source_prem_input = Input(shape=(self.max_len,), dtype='int32')
		source_hypo_input = Input(shape=(self.max_len,), dtype='int32')
		
		target_prem_input = Input(shape=(self.max_len,), dtype='int32')
		target_hypo_input = Input(shape=(self.max_len,), dtype='int32')
		
		# Look up Embeddings
		source_prem = source_embedding_layer(source_prem_input)
		source_hypo = source_embedding_layer(source_hypo_input)
		
		target_prem = target_embedding_layer(target_prem_input)
		target_hypo = target_embedding_layer(target_hypo_input)
			
		# LSTM Encoder for Source Language, Target Language, and for Both
		source_lstm_layer = Bidirectional(CuDNNLSTM(self.lstm_dim, return_sequences=True))
		target_lstm_layer = Bidirectional(CuDNNLSTM(self.lstm_dim, return_sequences=True))
		
		shared_lstm_layer = Bidirectional(CuDNNLSTM(self.lstm_dim, return_sequences=True))
		
		# Maxpooling LSTM encdoes vectors
		private_source_prem = GlobalMaxPooling1D()(source_lstm_layer(source_prem))
		private_source_hypo = GlobalMaxPooling1D()(source_lstm_layer(source_hypo))
		
		private_target_prem = GlobalMaxPooling1D()(target_lstm_layer(target_prem))
		private_target_hypo = GlobalMaxPooling1D()(target_lstm_layer(target_hypo))
		
		# Maxpooling Shared LSTM encodes vectors
		shared_source_prem = GlobalMaxPooling1D()(shared_lstm_layer(source_prem))
		shared_source_hypo = GlobalMaxPooling1D()(shared_lstm_layer(source_hypo))
		
		shared_target_prem = GlobalMaxPooling1D()(shared_lstm_layer(target_prem))
		shared_target_hypo = GlobalMaxPooling1D()(shared_lstm_layer(target_hypo))
		

		# Mergeing shared vectors and pass it to dense layer (adversarial training with gradient reversal layer)
		if self.is_adv:
			self.adv_loss = self.adversarial_loss(shared_source_prem, shared_source_hypo, to_categorical(np.zeros(self.batch_size),2)) \
								+ self.adversarial_loss(shared_target_prem, shared_target_hypo, to_categorical(np.ones(self.batch_size),2))
		
		# final representation: shared + private concatenated
		source_prem_encoded = Concatenate()([shared_source_prem, private_source_prem])
		source_hypo_encoded = Concatenate()([shared_source_hypo, private_source_hypo])
		
		target_prem_encoded = Concatenate()([shared_target_prem, private_target_prem])
		target_hypo_encoded = Concatenate()([shared_target_hypo, private_target_hypo])
		
		# Merging two LSTM encodes vectors from sentences
		source_merged = Concatenate()([source_prem_encoded, source_hypo_encoded, submult(source_prem_encoded,source_hypo_encoded)])	
		target_merged = Concatenate()([target_prem_encoded, target_hypo_encoded, submult(target_prem_encoded,target_hypo_encoded)])

		# final classifier for each task
		source_preds = self.classifier(source_merged)
		target_preds = self.classifier(target_merged)
		
		model = Model([source_prem_input, source_hypo_input, target_prem_input, target_hypo_input], [source_preds, target_preds])
		if self.is_adv:
			model.add_loss(self.adv_loss*self.lambda1)
		
		print(model.summary())
		model.compile(optimizer=self.optimizer, loss=['categorical_crossentropy','categorical_crossentropy'], metrics=['accuracy'])

		# Check point 
		checkpoint_dir = "models/" + str(int(time.time())) + '/'
		
		# mkdir if not exists
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		STAMP = 'lstm_%d_%d_%.1f' % (self.lstm_dim, self.dense_dim, self.drop_prob)
		
		filepath= checkpoint_dir + STAMP + "_%s_%s_{val_dense_8_acc:.2f}.h5"%(source_language, target_language)
		checkpoint = ModelCheckpoint(filepath, monitor='val_dense_8_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')

		lr_sched = ReduceLROnPlateau(monitor='val_dense_8_loss', factor=0.2, patience=1, cooldown=1, verbose=1)
		early_stopping = EarlyStopping(monitor='val_dense_8_acc', patience=10)
		
		model.fit([source_train_X, source_train_Y, target_train_X, target_train_Y], [source_train_Z, target_train_Z], 
				validation_data = ([source_val_X, source_val_Y, target_val_X, target_val_Y], [source_val_Z, target_val_Z]),
				epochs=self.epochs, batch_size = self.batch_size, shuffle = True,
				callbacks=[checkpoint, lr_sched, early_stopping])	

				
				
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--source_language', type=str, default='en', help='source_language')
	parser.add_argument('--target_language', type=str, default='es', help='target_language')
	parser.add_argument('--is_train', type=str, default=True, help='Training Mode')
	parser.add_argument('--is_adv', type=str, default=True, help='whether  apply adversarial training or not')
	params = parser.parse_args()
	
	setting = settings.Setting()

	Ours = Multimodel(setting, params.is_train, params.is_adv)
	Ours.train_model(params.source_language, params.target_language)				
