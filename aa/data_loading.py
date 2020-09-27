
#basics
import random
import pandas as pd
import torch
import os
import elementpath
import xml.etree.ElementTree as ET
from pathlib import Path
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

iddict = dict()

def get_tree_from_file(data_dir):
        list_to_df = []
        train = glob.glob("{}/*/*/*.xml".format(data_dir)) #returns te.
        test = glob.glob("{}/*/*/*/*.xml".format(data_dir))
        # file = [glob.glob("{}*.xml".format(item)) for item in directories] #returns equal many folders containing xml-files
        for item in train, test:
            list_to_df.append(item) #iterate and add to final folder to get format nested list with a list pr. directory containing xml files for furture DF-split.
        flat_list = [i for sublist in list_to_df for i in sublist]
        return flat_list


def make_pd_from_tree(lists_of_file_names):
        # print(lists_of_file_names)
        data_df = pd.DataFrame(columns = ["sentence_id", "token_id", "char_start_id", "char_end_id", "split"]) #for test purposes
        ner_df = pd.DataFrame(columns = ["sentence_id", "ner_id", "char_start_id", "char_end_id"]) # for test
        index = 1   
        for filename in lists_of_file_names:
            if 'Test' in filename:
                split = 'test'
            else:
                split = 'train'
            xml = ET.parse(filename)
            root = xml.getroot()
            for sentence in root.iter('sentence'): # get sent id and text 
                sent_id = sentence.get('id')
                # print(sentence.attrib)
                for item in sentence:
                    # print(item.attrib)
                    if item.tag == 'entity': #ensuring that ther e is an entity at all. would otherwise get 000000 values and spoil the model.
                        ner_id = item.get('type') #ner = type
                        token_id = item.get('text') #entity = name / text
                        if ";" in item.get('charOffset'):
                            char_offsets = item.get('charOffset').split(';') #split charoffset
                            for span in char_offsets:
                                char_start_id, char_end_id = span.split('-')
                                # print(char_start_id, char_end_id)
                                data_df.loc[index] = [sent_id, token_id, char_start_id, char_end_id, split]
                                ner_df.loc[index] = [sent_id, ner_id, char_start_id, char_end_id]
                                index += 1
                                # print(data_df, ner_df)
                                # break
                        else: 
                            char_start_id, char_end_id = item.get('charOffset').split('-')[0], item.get('charOffset').split('-')[1] # split char off set
                            # print(char_start_id, char_end_id)
                            data_df.loc[index] = [sent_id, token_id, char_start_id, char_end_id, split]
                            ner_df.loc[index] = [sent_id, ner_id, char_start_id, char_end_id]
                            # print(data_df, ner_df)
                            index += 1     
            # if index > 10: # to get smaller dataset, only for tester. 
            #     break   
        return data_df, ner_df

# data_frames = make_pd_from_tree(get_tree_from_file('DDICorpus'))
# print(data_frames[0])
# print('---')
# print(data_frames[1])

data_df = make_pd_from_tree(get_tree_from_file('DDICorpus'))[0]
ner_df = make_pd_from_tree(get_tree_from_file('DDICorpus'))[1]
# print('printing data df with labels')
# print(data_df)

# print(ner_df['ner_id'].value_counts())

#functions with pseudo returns instead of .self.
def data_df_label_encoding(data_df):
    #label_encoding
    lb_make_data = LabelEncoder()
    data_df['token_id'] = lb_make_data.fit_transform(data_df['token_id'])
    lb_make_name_mapping_data = dict(zip(lb_make_data.classes_, lb_make_data.transform(lb_make_data.classes_)))
    # print(data_df)
    return data_df, lb_make_name_mapping_data

# print('Printing label encoded data df')
# print(data_df_label_encoding(data_df)[1])

#functions with pseudo returns instead of .self.
def ner_id_label_encoding(ner_df):
    #label_encoding
    lb_make_ner = LabelEncoder()
    ner_df['ner_id'] = lb_make_ner.fit_transform(ner_df['ner_id'])
    lb_make_name_mapping_ner = dict(zip(lb_make_ner.classes_, lb_make_ner.transform(lb_make_ner.classes_)))
    # print(ner_df)
    return ner_df, lb_make_name_mapping_ner

# print(ner_id_label_encoding(ner_df))


#setting variable for de-coding variables: --> must be done before train/test splitting
id2word = data_df_label_encoding(data_df)[1]
id2ner = ner_id_label_encoding(ner_df)[1]


#making split/validation + test set of data_df
#made with functions instead of .self.
train, validate = np.split(data_df_label_encoding(data_df)[0].loc[data_df['split'] == 'train'].sample(frac=1), [int(.2*len(data_df_label_encoding(data_df)[0]))])
# print(train,validate)


#make test. only testable if i = total
test = data_df_label_encoding(data_df)[0].loc[data_df['split'] == 'test']
# print('printing test-data from data df')
# print(test)


class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)


    def get_tree_from_file(self, data_dir):
        list_to_df = []
        train = glob.glob("{}/*/*/*.xml".format(data_dir)) #returns te.
        test = glob.glob("{}/*/*/*/*.xml".format(data_dir))
        # file = [glob.glob("{}*.xml".format(item)) for item in directories] #returns equal many folders containing xml-files
        for item in train, test:
            list_to_df.append(item) #iterate and add to final folder to get format nested list with a list pr. directory containing xml files for furture DF-split.
        flat_list = [i for sublist in list_to_df for i in sublist]
        pass


    def make_pd_from_tree(self, lists_of_file_names):
        # print(lists_of_file_names)
        # data_df = pd.DataFrame(columns = ["sent_id", "token_id", "char_start_id", "char_end_id", "split"]) #for test purposes
        # ner_df = pd.DataFrame(columns = ["sent_id", "ner_id", "char_start_id", "char_end_id"]) # for test
        self.id2ner = dict()
        self.id2word = dict()
        index = 1   
        for filename in lists_of_file_names:
            if 'Test' in filename:
                split = 'test'
            else:
                split = 'train'
            xml = ET.parse(filename)
            root = xml.getroot()
            for sentence in root.iter('sentence'): # get sent id and text 
                sent_id = sentence.get('id')
                # print(sentence.attrib)
                for item in sentence:
                    # print(item.attrib)
                    if item.tag == 'entity': #ensuring that ther e is an entity at all. would otherwise get 000000 values and spoil the model.
                        ner_id = item.get('type') #ner = type
                        token_id = item.get('text') #entity = name / text
                        if ";" in item.get('charOffset'):
                            char_offsets = item.get('charOffset').split(';') #split charoffset
                            for span in char_offsets:
                                char_start_id, char_end_id = span.split('-')
                                # print(char_start_id, char_end_id)
                                self.data_df.loc[index] = [sent_id, token_id, char_start_id, char_end_id, split]
                                self.ner_df.loc[index] = [sent_id, ner_id, char_start_id, char_end_id]
                                index += 1
                                # print(data_df, ner_df)
                                # break
                        else: 
                            char_start_id, char_end_id = item.get('charOffset').split('-')[0], item.get('charOffset').split('-')[1] # split char off set
                            # print(char_start_id, char_end_id)
                            self.data_df.loc[index] = [sent_id, token_id, char_start_id, char_end_id, split]
                            self.ner_df.loc[index] = [sent_id, ner_id, char_start_id, char_end_id]
                            # print(data_df, ner_df)
                            index += 1     
            if index > 10: # to get smaller dataset, only for tester. 
                break   

        pass


    def data_df_label_encoding(self):
        #label_encoding
        lb_make_df = LabelEncoder()
        self.data_df['token_id'] = lb_make_df.fit_transform(data_df['token_id'])
        self.lb_make_name_mapping = dict(zip(lb_make_df.classes_, lb_make_df.transform(lb_make_df.classes_)))
        self.id2word = self.lb_make_name_mapping
        # print(data_df)
        pass


    def ner_id_label_encoding(self):
        #label_encoding
        lb_make_ner = LabelEncoder()
        self.ner_df['ner_id'] = lb_make_ner.fit_transform(ner_df['ner_id'])
        self.lb_make_name_mapping = dict(zip(lb_make_ner.classes_, lb_make_ner.transform(lb_make_ner.classes_)))
        self.id2ner = self.lb_make_name_mapping
        # print(data_df)
        pass


    def _parse_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        
        self.data_df = pd.DataFrame(columns = ["sentence_id", "token_id", "char_start_id", "char_end_id", "split"]) #for test purposes
        self.ner_df = pd.DataFrame(columns = ["sentence_id", "ner_id", "char_start_id", "char_end_id"]) # for test


        #fill out the DF's above.
        make_pd_from_tree(get_tree_from_file('DDICorpus'))
        data_frames = make_pd_from_tree(get_tree_from_file('DDICorpus'))

        print('Printing Data_df', data_frames[0])
        print('---')
        print('Printing Ner_df', data_frames[1])

        #Label-Encode text-values in DF
        print('Encoding labels...')
        data_df_label_encoding()
        ner_id_label_encoding()

        #split sets --- added .sample(frac=1) to shuffle the rows of the selected part of the DF.
        #the two respective training and validation set has been divided 8/2
        train, validate= np.split(self.data_df.loc[self.data_df['split'] == 'train'].sample(frac=1), [int(.2*len(self.data_df))])
        test = self.data_df.loc[self.data_df['split'] == 'test']
        print('Making train and validation set in ratio 8-2...')







        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.

        pass


    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        pass

    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        pass

