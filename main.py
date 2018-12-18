import numpy as np
from settings import *
from data_loader import *
from keras.layers import *
from keras.activations import softmax
from keras.regularizers import l2
from keras.models import *
import pandas as pd
from keras.optimizers import Adam
from flipGradientTF import *


source_embeddings = Embedding(len(word_dict_source), word_embed_size, weights=[source_vectors], input_length=(max_len_sentence,), trainable=False)

s1_input1 = Input(shape=(max_len_sentence,))
s1_input2 = Input(shape=(max_len_sentence,))

s2_input1 = Input(shape=(max_len_sentence,))
s2_input2 = Input(shape=(max_len_sentence,))


s1_tasklabel = Input(shape=(2,))
s2_tasklabel = Input(shape=(2,))


s1_q1 = source_embeddings(s1_input1)
s1_q2 = source_embeddings(s1_input2)

s2_q1 = source_embeddings(s2_input1)
s2_q2 = source_embeddings(s2_input2)


s1_lstm = Bidirectional(CuDNNLSTM(lstm_dimension, return_sequences=True))
s2_lstm = Bidirectional(CuDNNLSTM(lstm_dimension, return_sequences=True))
shared_lstm = Bidirectional(CuDNNLSTM(lstm_dimension, return_sequences=True))

s1_q1_private = GlobalMaxPooling1D()(s1_lstm(s1_q1)) 
s1_q2_private = GlobalMaxPooling1D()(s1_lstm(s1_q2))

s2_q1_private = GlobalMaxPooling1D()(s2_lstm(s2_q1))
s2_q2_private = GlobalMaxPooling1D()(s2_lstm(s2_q2))


s1_q1_shared = GlobalMaxPooling1D()(shared_lstm(s1_q1)) 
s1_q2_shared = GlobalMaxPooling1D()(shared_lstm(s1_q2))
s2_q1_shared = GlobalMaxPooling1D()(shared_lstm(s2_q1)) 
s2_q2_shared = GlobalMaxPooling1D()(shared_lstm(s2_q2))


def adversarial(merged):
	dense = Dense(dense_dim, activation='tanh')(merged)
	dense = Dropout(dropout_rate)(dense)
	out = Dense(2, activation='softmax', name="adversarial")(dense)
	return out

Flip = GradientReversal(1)
s1_q1_fliped = Flip(s1_q1_shared)
s1_q2_fliped = Flip(s1_q2_shared)
s2_q1_fliped = Flip(s2_q1_shared)
s2_q2_fliped = Flip(s2_q2_shared)

s1_shared = Concatenate()([s1_q1_fliped, s1_q2_fliped, submult(s1_q1_fliped, s1_q2_fliped)])
s2_shared = Concatenate()([s2_q1_fliped, s2_q2_fliped, submult(s2_q1_fliped, s2_q2_fliped)])
	
shared_s1_out = adversarial(s1_shared)
shared_s2_out = adversarial(s2_shared)	

adv_loss = K.mean(K.categorical_crossentropy(s1_tasklabel, shared_s1_out)) + K.mean(K.categorical_crossentropy(s2_tasklabel, shared_s2_out))


s1_q1_encoded = Concatenate()([s1_q1_shared, s1_q1_private])
s1_q2_encoded = Concatenate()([s1_q2_shared, s1_q2_private])
s2_q1_encoded = Concatenate()([s2_q1_shared, s2_q1_private])
s2_q2_encoded = Concatenate()([s2_q2_shared, s2_q2_private])

	
s1_merged = Concatenate()([s1_q1_encoded, s1_q2_encoded, submult(s1_q1_encoded, s1_q2_encoded)])
s2_merged = Concatenate()([s2_q1_encoded, s2_q2_encoded, submult(s2_q1_encoded, s2_q2_encoded)])


def classifier(merged):
	dense = Dense(lstm_dimension, activation='tanh')(merged)
	dense = Dropout(dropout_rate)(dense)
	out = Dense(3, activation='softmax')(dense)
	return out

def classifier2(merged):
	dense = Dense(lstm_dimension, activation='tanh')(merged)
	dense = Dropout(dropout_rate)(dense)
	out = Dense(3, activation='softmax')(dense)
	return out	
	
	
s1_out = classifier(s1_merged)
s2_out = classifier2(s2_merged)


model = Model([s1_input1,s1_input2,s1_tasklabel,s2_input1,s2_input2,s2_tasklabel], [s1_out, s2_out])
model.add_loss(adv_loss*lambda1)

print(model.summary())
model.compile(optimizer=Adam(0.0005), loss=['categorical_crossentropy','categorical_crossentropy'], metrics=['accuracy'])


model.fit([source_trainX, source_trainY, source_trainL, source2_trainX, source2_trainY, source2_trainL], [source_trainZ, source2_trainZ], 
		validation_data = ([source_devX, source_devY, source_devL, source2_devX, source2_devY, source2_devL], [source_devZ, source2_devZ]),
		batch_size = 1024,
		shuffle = True,
		epochs=70)
