import utils
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional,Dropout
from keras.preprocessing.sequence import pad_sequences


def get_labeled_comments(df):
    mask = (df['toxic'] == 1) | (df['severe_toxic'] == 1) | (df['obscene'] == 1) | \
           (df['threat'] == 1) | (df['insult'] == 1) | (df['identity_hate'])
    return df[mask].drop('id', 1)


def load_vocab():
    word2vec = utils.Word2Vec.load('models/word2Vec_model')
    return {k: v for k, v in zip(word2vec.wv.index2word, range(0, len(word2vec.wv.index2word)))}


def get_encoded_comments(x_train, vocab):
    encoded_comments = utils.get_encoded_comments(utils.comments_to_token(x_train.tolist()), vocab)
    return pad_sequences(encoded_comments, maxlen=100, padding='post')


def build_model(vocab_size, embedding_matrix):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=100, trainable=False))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model


def get_embedding_matrix(vocab_size):
    word2vec = utils.Word2Vec.load('models/word2Vec_model')
    w2v = dict(zip(word2vec.wv.index2word, word2vec.wv.syn0))
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in vocab.items():
        embedding_vector = w2v.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


if __name__ == '__main__':
    train_raw = pd.read_csv('data/train.csv')
    train = get_labeled_comments(train_raw)
    vocab = load_vocab()
    vocab_size = len(vocab)
    X_train, X_test, y_train, y_test = train_test_split(train['comment_text'], train[['toxic', 'severe_toxic',
                                                        'obscene', 'threat', 'insult', 'identity_hate']], test_size=0.2,
                                                       shuffle=True)
    comments = get_encoded_comments(X_train, vocab)
    embedding_matrix = get_embedding_matrix(vocab_size)
    model = build_model(vocab_size, embedding_matrix)

    model.fit(comments, y_train , epochs=10, verbose=0)

    model.save("models/models.h5")

    # evaluate the model
    comments = get_encoded_comments(X_test, vocab)
    loss, accuracy = model.evaluate(comments, y_test.tolist(), verbose=0)
    print('Accuracy: %f' % (accuracy * 100))

