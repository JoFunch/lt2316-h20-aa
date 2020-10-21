
#basics
import pandas as pd
import nltk
import numpy as np
import torch

def int_to_word(word, id2word):
    return id2word[word]

def uppercase(sent):
	uppercase = []
	if sent:
		if sent.isupper():
			uppercase.append(1)
		else:
			uppercase.append(0)
	return uppercase

# print(uppercase(item))


def pos_tag(sent):
	pos_tag = []
	tagged = nltk.pos_tag(sent)
	return tagged[1]



def add_features_to_df(data, id2word):
# Feel free to add any new code to this script
	#make columns
	data['word'] = data.loc[:,'token_id'].apply(int_to_id)

	data['pos_tag'] = data.loc[:, 'token_id'].apply(pos_tag)

	data['uppercase'] = data.loc[:, 'token_id'].apply(lambda x: 1 if x.isupper() else 0)
	print(data)
	return data
    

def pos_tag_encoding(df):
        #label_encoding
        lb_make_df = LabelEncoder()
        df['pos_tag'] = lb_make_df.fit_transform(df['pos_tag'])
        lb_make_df_name_mapping = dict(zip(lb_make_df.classes_, lb_make_df.transform(lb_make_df.classes_)))
        id2pos = lb_make_df_name_mapping
        # print(data_df)
        return df, id2pos    
    
    
    
    

    
def encode_new_features(data):
    pos_tag_encoding(data)

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
