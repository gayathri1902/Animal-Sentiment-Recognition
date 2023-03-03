import pandas as pd
import librosa
import os
import numpy as np

def extract_features(file):
    audio,sr = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features
paths=[]
labels=[]
extracted_features=[]

# data=extract_features("dataset/angry/angry 1.mp3")
#print(data)
os.chdir('dataset/')

extracted_features=[]

for i in range(0,12):
    filename="angry/angry "+str(i+1)+".mp3"
    label="angry"
    data=extract_features(filename)
    extracted_features.append([data,label])
print(extracted_features[5])

for i in range(0,12):
    filename="sad/sad "+str(i+1)+".mp3"
    label="sad"
    data=extract_features(filename)
    extracted_features.append([data,label])
print(extracted_features[2])


ex_ft_df=pd.DataFrame(extracted_features,columns=['features','class'])
ex_ft_df.head()

X=np.array(ex_ft_df['features'].tolist())
y=np.array(ex_ft_df['class'].tolist())
print(X.shape)
print(y)


#import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

print(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape,"\n",X_test.shape,"\n",y_train.shape,"\n",y_test.shape)


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
num_labels=y.shape[1]
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

## Trianing my model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)
test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])




filename="test4.mp3"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)
predictedl1=model.predict(mfccs_scaled_features)
predicted_label=np.argmax(predictedl1,axis=1)


print(predicted_label)
prediction_class = labelencoder.inverse_transform(predicted_label) 
print("PREDICTION: ",prediction_class)
