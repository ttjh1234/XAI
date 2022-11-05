import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def imdb_model(X_train,y_train,path,embedding_dim=100,hidden_units=128,vocab_size=10000):
    
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(GRU(hidden_units,return_sequences=True))
    model.add(GRU(hidden_units))
    model.add(Dense(1, activation='sigmoid'))

    es = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    mc = ModelCheckpoint('GRU_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=30, callbacks=[es, mc], batch_size=64, validation_split=0.2)

    model.save(path)


def imdb_data_fetch(vocab_size=10000,max_len=500):
    
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
    word_index = tf.keras.datasets.imdb.get_word_index()
    id_to_word={id_ + 3 : word for word,id_ in word_index.items()}

    for id_,token in enumerate(["<pad>","<sos>","<unk>"]):
        id_to_word[id_]=token
        
    return (X_train,y_train), (X_test,y_test), word_index, id_to_word

def pretrained_model(path):

    model=tf.keras.models.load_model(path)
    print(model.summary())
    
    return model


