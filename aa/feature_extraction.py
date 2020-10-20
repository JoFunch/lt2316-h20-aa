
#basics
import pandas as pd
import nltk
import numpy as np
import torch



def uppercase(sent):
	uppercase = []
	for word in sent:
		if word.isupper():
			uppercase.append(1)
		else:
			uppercase.append(0)
	return uppercase

# print(uppercase(item))


def pos_tag(sent):
	pos_tag = []
	tagged = nltk.pos_tag(sent)
	for word, tag in tagged:
		pos_tag.append(tag)
		# print(word, tag)


	return pos_tag



def add_features_to_df(data, id2word):
# Feel free to add any new code to this script
	#make columns
	data['word'] = data.loc['sentence_id'].apply(int2id)

	data['pos_tag'] = data.loc['sentence_id'].apply(pos_tag)

	data['uppercase'] = data.loc['sentence_id'].apply(uppercase)


	


	pass

def extract_features(data:pd.DataFrame, max_sample_length:int):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    pass
