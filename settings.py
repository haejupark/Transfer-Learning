import sys

class Setting(object):
	def __init__(self):
		self.lr = 0.001
		self.word_dim = 300
		self.lstm_dim = 256
		self.max_len = 30
		self.dense_dim =  128
		self.keep_prob = 0.6
		self.batch_size = 1024
		self.epochs = 50
		self.lambda1 = 0.005
		self.train_dir = ""
		self.dev_dir = ""
		self.embed_dir = ""

labels = {'neutral':0, 'entailment':1, 'contradiction':2}
labels_reverse = {0:'neutral', 1:'entailment', 2:'contradiction'}

