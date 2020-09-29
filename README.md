# LT2316 H20 Assignment A1

Jonas Funch

## Notes on Part 1.

The pipeline of accessing the XML-files, processing them, and plotting them was diveded into several smaller funcitons for overview and transparency.

F1. Access and allign everything in folders to a flat list. 

F2. Iterate the file-list. Preprocessing is as follows: remove any word-end-punctuation as keeping it would decrease accuraccy / duplicate data/input. Also, any tokenized entity being entirely non-alphabetical has been removed from the data. 

F2. Data-df consists of all entries from the corpus with their respective start and end character as index of the respective sentence. The index had to be forced to type(int) for it to fit the get-Random-sample function.

F2 likewith in Ner_id but with NERs and their group. 

F2 restricted index for shorter running time. 

F3-4: Labelencoding. Although the coding process is significantly shorter than writing/integrating dictionary-look-ups for turning IDs into integers/vice-versa, the labelencoder proved much useful. 
It selects the given DF and returns unique ints for entry. Assigned to the self.id2ner for decoding purposes. 

Parse-data: instanciating dataframes, filling them out, and printing them individually, both before and after label encoding.
Creating Vocab of decoded entries. 
turning keys and values of id2word/ner in order for them to work with get_random_sample. 

Making test, val and train set where val and train are randomized through "sample(trac=1)" to a ratio of 80/20 in train/val. 

Max-length set to 70.





## Notes on Part 2.

*fill in notes and documentation for part 2 as mentioned in the assignment description*

## Notes on Part Bonus.

*fill in notes and documentation for the bonus as mentioned in the assignment description, if you choose to do the bonus*
