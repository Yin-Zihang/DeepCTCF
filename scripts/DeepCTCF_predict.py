
### Load arguments

import sys, getopt
def main(argv):
   new_seq = ''
   model_ID = ''
   try:
      opts, args = getopt.getopt(argv,"hs:m:",["seq=","model="])
   except getopt.GetoptError:
      print('DeepSTARR_pred_new_sequence.py -s <fasta seq file> -m <CNN model file>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('DeepSTARR_pred_new_sequence.py -s <fasta seq file> -m <CNN model file>')
         sys.exit()
      elif opt in ("-s", "--seq"):
         new_seq = arg
      elif opt in ("-m", "--model"):
         model_ID = arg
   if new_seq=='': sys.exit("fasta seq file not found")
   if model_ID=='': sys.exit("CNN model file not found")
   print('Input FASTA file is ', new_seq)
   print('Model file is ', model_ID)
   return new_seq, model_ID

if __name__ == "__main__":
   new_seq, model_ID = main(sys.argv[1:])



### Load libraries

import tensorflow as tf
import os
import re
import numpy as np
import pandas as pd
import math                       
from scipy import stats
from scipy.stats import shapiro
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

### Load sequences
print("\nLoading sequences ...\n")
seqName={}
sequences=[]
for eachLine in open(new_seq):
	if '>' not in eachLine:
		sequence=eachLine.split()[0]
		sequences.append(sequence)
		seqName[sequence]=name
	else:
		name=eachLine.split()[0].split('>')[1]

print('%s sequences loaded'%len(sequences))
basecat=[0,1,2,3]
# The LabelEncoder encodes a sequence of bases as a sequence of integers.
integer_encoder = LabelEncoder()  
integer_encoder.fit(["A", "C", "G", "T"])
# The OneHotEncoder converts an array of integers to a sparse matrix where 
# each row corresponds to one possible value of each feature.
one_hot_encoder = OneHotEncoder(categories=[basecat])

input_features = []
test_features=[]
for sequence in sequences:
	integer_encoded = integer_encoder.transform(list(sequence))#;print(integer_encoded)
	integer_encoded = np.array(integer_encoded).reshape(-1, 1)#;print(integer_encoded) to 1 column 2d arrary
	one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)#;print(integer_encoded)
	test_features.append(one_hot_encoded.toarray())

test_features = np.stack(test_features)

#load model
model = tf.keras.models.load_model(model_ID)
print('model is loaded')
test_labels_ratio=model.predict(np.stack(test_features))
print('predicted')

pred={}
for i in range(test_features.shape[0]):
	dnacode=''
	for j in range(42):
		if list(test_features[i,j])==[1,0,0,0]:
			dnacode+='A'
		if list(test_features[i,j])==[0,1,0,0]:
			dnacode+='C'
		if list(test_features[i,j])==[0,0,1,0]:
			dnacode+='G'
		if list(test_features[i,j])==[0,0,0,1]:
			dnacode+='T'


	if dnacode in sequences:
		pred[dnacode]=test_labels_ratio[i,][0]

print("\nSaving file ...\n")

fjob=open('DeepCTCF_predicted.csv','w')
fjob.write('Sequence_Name,Sequence,Predicted_Affinity\n')

count=0
for sequence in sequences:
	affinity=pred[sequence]
	name=seqName[sequence]
	fjob.write(f'{name},{sequence},{affinity}\n')
fjob.close()

