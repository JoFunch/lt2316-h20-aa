
#basics
import pandas as pd
import nltk
from nltk import pos_tag#_sents
#nltk.download('averaged_perceptron_tagger')
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

#def int_to_word(word, id2word):
 #   return id2word[word]




#def pos_tag(sent):
	#pos_tag = []
	#tagged = nltk.pos_tag(sent)
	#return tagged[1]



def add_features_to_df(data, id2word):
    #Feel free to add any new code to this script
    #make columns
    data['word'] = data.loc[:,'token_id'].apply(lambda x: [id2word[x]])
    data['pos_tag'] = data.loc[:, 'word'].apply(pos_tag) #_sents)
    data['pos_tag'] = data['pos_tag'].apply(lambda x: x[0][1])
    data['word'] = data.loc[:,'word'].apply(lambda x: x[0])
    data['uppercase'] = data.loc[:, 'word'].apply(lambda x: x[0].isupper())
    data['uppercase'] = data.loc[:, 'uppercase'].apply(lambda x: 1 if x == True else 0)
    data['length'] = data.loc[:, 'word'].apply(lambda x: len(x))
    #print(data)
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
    encoding_df = pos_tag_encoding(data)[0]
    encoding_dict = pos_tag_encoding(data)[1]

    return encoding_df, encoding_dict

def df_to_tens(df_split):
        """
        The idea here is to convert a DF-split into a list of sorts and loop it rhough the NER to see if any hits.

        """
        y_tensor = []

        sent_id = df_split.groupby('sentence_id')
        # print(merged_df)
        for sentence_id, df_grouped_by_sentence in merged:
            df_sent = [int(v) for v in list(df_grouped_by_sentence['ner_id'])]
            # print(df_sent)
            if len(df_sent) < self.max_sample_length: #pad
                self.padding(self.max_sample_length, df_sent)
                y_tensor.append(df_sent)
            else:
                if len(df_sent) >= self.max_sample_length:
                    y_tensor.append(df_sent)
        y_tensor = np.asarray(y_tensor, dtype=np.float32)
                    
        return y_tensor






def extract_features(data:pd.DataFrame, max_sample_length:int, id2word):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    
    
    #Get Features into DF
    features = add_features_to_df(data, id2word)
    #print(features)
    encoded_features = encode_new_features(features)[0]
    #print(encoded_features)
    
    
    #Split DF with features in it
    train, validate= np.split(encoded_features.loc[self.data_df['split'] == 'train'].sample(frac=1), [int(.2*len(encoded_features))])
    #test = self.data_df.loc[self.data_df['split'] == 'test']
    
    
    #make df_split into arrays
    
    
    
    
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    return encoded_features
