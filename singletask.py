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

np.random.seed(0)

class NLImodel(object):
	def __init__(self,setting,is_train=True):
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
		#self.is_adv = is_adv
		self.is_train = is_train
		
		self.train_dir = setting.train_dir
		self.dev_dir = setting.dev_dir
		self.embed_dir = setting.embed_dir

	def train_model(self, source_language):
		
		train_name = self.train_dir+"multinli.train.%s.txt"%source_language
		dev_name = self.dev_dir+"xnli_%s.txt"%source_language
		embeddings_name = self.embed_dir+"wiki.%s.vec"%source_language
		
		train_data=[l.strip().split('\t') for l in open(train_name, errors='ignore')]
		dev_data=[l.strip().split('\t') for l in open(dev_name, errors='ignore')]

		word_vocab = get_vocab(train_data + dev_data)	
		word_embeddings = get_embeddings(word_vocab, embeddings_name, setting.word_dim)

		train_X, train_Y, train_Z, \
			val_X, val_Y, val_Z	= create_train_dev_set(train_data, dev_data, word_vocab)
		
		if train_X is None:
			print("++++++ Unable to train model +++++++")
			return None	

			
		embedding_layer = Embedding(len(word_vocab), self.word_dim, weights=[word_embeddings], input_length=(self.max_len,), trainable=False)
		
		prem_input = Input(shape=(self.max_len,), dtype='int32')
		hypo_input = Input(shape=(self.max_len,), dtype='int32')
		
		prem = embedding_layer(prem_input)
		hypo = embedding_layer(hypo_input)
		
		lstm_layer = Bidirectional(CuDNNLSTM(self.lstm_dim, return_sequences=True))
		
		prem = GlobalMaxPooling1D()(lstm_layer(prem))
		hypo = GlobalMaxPooling1D()(lstm_layer(hypo))
		
		merged = Concatenate()([prem, hypo, submult(prem,hypo)])

		dense = Dropout(self.drop_prob)(merged)
		dense = Dense(self.dense_dim, kernel_initializer='uniform', activation='relu')(dense)
		dense = Dropout(self.drop_prob)(dense)
		preds = Dense(3, activation='softmax')(dense)
			
		model = Model([prem_input, hypo_input], [preds])
		model.compile(optimizer=self.optimizer, loss=['categorical_crossentropy'], metrics=['accuracy'])
		
		checkpoint_dir = "models/" + str(int(time.time())) + '/'
		
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		STAMP = 'lstm_%d_%d_%.1f' % (self.lstm_dim, self.dense_dim, self.drop_prob)
		
		filepath= checkpoint_dir + STAMP + "_%s_{val_acc:.2f}.h5"%source_language
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')

		lr_sched = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, cooldown=1, verbose=1)
		early_stopping = EarlyStopping(monitor='val_acc', patience=10)
		
		model.fit([train_X, train_Y], [train_Z], 
				validation_data = ([val_X, val_Y], [val_Z]),
				epochs=self.epochs, batch_size = self.batch_size, shuffle = True,
				callbacks=[checkpoint, lr_sched, early_stopping])	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--source_language', type=str, default='es', help='source_language')
	params = parser.parse_args()
	
	setting = settings.Setting()

	Tars = NLImodel(setting)
	Tars.train_model(params.source_language)
