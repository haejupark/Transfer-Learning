import sys

word_embed_size = 300
max_len_sentence = 50

cnli_train_path = "data/cnli/cnli_train_processed.txt"
cnli_dev_path = "data/cnli/cnli_dev_processed.txt"
#cnli_test_path = "data/cnli/cnli_test_processed.txt"

snli_train_path = "data/snli/train_processed.txt"
snli_dev_path = "data/snli/dev_processed.txt"

embeddings_chinese = "data/sgns.merge.word" 
embeddings_english = "data/wiki.en.vec"

labels = {'neutral':0, 'entailment':1, 'contradiction':2}
labels_reverse = {0:'neutral', 1:'entailment', 2:'contradiction'}
