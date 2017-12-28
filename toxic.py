
import utils
import numpy as np
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

train_raw = pd.read_csv('train.csv')

data = train_raw['comment_text'].tolist()

tokenized_coments = utils.comments_to_token(data)

word2vec = utils.get_word2vec(tokenized_coments)

w2v = dict(zip(word2vec.wv.index2word, word2vec.wv.syn0))

vocab = {k : v for k,v in zip(word2vec.wv.index2word, range(0, len(word2vec.wv.index2word)))}

vector_len = 100
vocab_size = len(vocab)


embedding_matrix = np.zeros((vocab_size, 100))
for word, i in vocab.items():
    embedding_vector = w2v.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

train_raw['foul'] = train_raw.apply(utils.is_foul, axis=1)
train_not_foul = train_raw[['comment_text','foul']][train_raw['foul'] == 0]
train_foul = train_raw[['comment_text','foul']][train_raw['foul'] == 1]

# Down sample not foul to 30000
train_not_foul = train_not_foul.sample(30000)

train = pd.concat([train_foul, train_not_foul])

X_train, X_test, y_train, y_test = train_test_split(train['comment_text'], train['foul'], test_size=0.2,
                                                    shuffle=True)


encoded_comments = utils.get_encoded_comments(utils.comments_to_token(X_train.tolist()), vocab)
padded_comments = pad_sequences(encoded_comments, maxlen=100, padding='post')

model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=100, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_comments,  y_train.tolist(), epochs=10, verbose=0)


# evaluate the model
encoded_comments = utils.get_encoded_comments(utils.comments_to_token(X_test.tolist()), vocab)
padded_comments = pad_sequences(encoded_comments, maxlen=100, padding='post')
loss, accuracy = model.evaluate(padded_comments, y_test.tolist(), verbose=0)
print('Accuracy: %f' % (accuracy*100))