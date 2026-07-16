import re
import keras
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout, BatchNormalization, Activation

sequence_length = 42
sequences=[]
labels=[]

for eachLine in open('oligos_train.txt'):
	label=eachLine.split()[0]
	seq=eachLine.split()[1]
	sequences.append(seq)
	labels.append(float(eachLine.split()[0]))
	if len(seq) !=42:
		print(eachLine,'error')
print(len(sequences))
print(len(labels))

# Let's print the first few sequences.
print(pd.DataFrame(sequences, index=np.arange(1, len(sequences)+1), 
             columns=['Sequences']).head())

# ensure labels to be 1d array
labels = np.array(labels)  
print('Labels:\n',labels.T)

#basecat=['A','C','G','T']
basecat=[0,1,2,3]


# The LabelEncoder encodes a sequence of bases as a sequence of integers.
integer_encoder = LabelEncoder()  
integer_encoder.fit(["A", "C", "G", "T"])

# The OneHotEncoder converts an array of integers to a sparse matrix where 
# each row corresponds to one possible value of each feature.
one_hot_encoder = OneHotEncoder(categories=[basecat])   
input_features = []
for sequence in sequences:
	integer_encoded = integer_encoder.transform(list(sequence))
	integer_encoded = np.array(integer_encoded).reshape(-1, 1)
	one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
	input_features.append(one_hot_encoded.toarray())

np.set_printoptions(threshold=40)
input_features = np.stack(input_features)
print("Example sequence\n-----------------------")
print('DNA Sequence #1:\n',sequences[0][:10],'...',sequences[0][-10:])
print('One hot encoding of Sequence #1:\n',input_features[0].T)

# define train and test set
train_features, test_features, train_labels, test_labels = train_test_split(
    input_features, labels, test_size=0.25, random_state=42
)
print('test_labels:\n',test_labels)
print("Train features shape:", train_features.shape)
print("Train labels shape:", train_labels.shape)
print("Test features shape:", test_features.shape)
print("Test labels shape:", test_labels.shape)

# build model

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(sequence_length, 4)))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=128, kernel_size=10, activation='relu')) 
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=128, kernel_size=4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.3)))

model.add(Dropout(0.2))
model.add(Dense(1))  
# compile model

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(loss='mean_squared_error', optimizer=optimizer)
# print model architecture
model.summary()

class R2Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super(R2Callback, self).__init__()  
        self.val_data = val_data  

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.val_data
        y_pred = self.model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        logs['val_r2'] = r2  


# prepare validation set
val_data = (test_features, test_labels)
#  add r2 callback
r2_callback = R2Callback(val_data)

#early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# train the model
history = model.fit(train_features, train_labels,
                    epochs=50,batch_size=32, validation_split=0.25,
                    callbacks=[early_stopping, r2_callback],
                    verbose=0)

# save the model
model.save(r'./example.h5')

# loss curve
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss (k41_f32)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('loss.png')

# predict on test set
test_predictions = model.predict(test_features)
mse = mean_squared_error(test_labels, test_predictions)
r2 = r2_score(test_labels, test_predictions)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
plt.figure()
plt.plot(history.history['val_r2'], label='validation R²')
plt.title('Model R² Score (k41_f32)')
plt.ylabel('R² Score')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('Mean Squared Error.png')


# scatter plot
test_predictions = model.predict(test_features).flatten()
scc, _ = spearmanr(test_labels, test_predictions)
print(f"Spearman Correlation Coefficient (SCC): {scc}")
plt.figure(figsize=(8, 6))
plt.scatter(test_labels, test_predictions, alpha=0.5, label='Predicted vs Actual')
plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], color='green', linestyle='--', label='Ideal Line')

# add fit curve
z = np.polyfit(test_labels, test_predictions, 1)
p = np.poly1d(z)
plt.plot(test_labels, p(test_labels), color='red', linestyle='--', label=f'SCC: {scc:.2f}')
plt.title('Predicted vs Actual Values with SCC Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.savefig('Spearman Correlation.png')

# Residuals vs test_labels
residuals = test_labels - test_predictions
plt.figure(figsize=(8, 6))
plt.scatter(test_labels, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')  
plt.title('Residuals vs test_labels')
plt.xlabel('test_labels')
plt.ylabel('Residuals')
plt.legend(['Residuals', 'Zero Line'])
plt.savefig('Residuals.png')
