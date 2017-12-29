import re
import h5py
import nltk
import os.path
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize, word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Bidirectional,Dropout


def tokenize_sentences(data):
    tokenized_sentences = []
    for comment in data:
        sentences = sent_tokenize(comment.strip())
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            cleaned_tokes = []
            for token in tokens:
                cleaned_tokes.append(re.sub('[^A-Za-z0-9]+', '', token).strip().lower())
            tokenized_sentences.append(list(filter(None, cleaned_tokes)))
    return tokenized_sentences


def get_sentence_len_stat(sentences):
    arr = np.asarray([len(sentence) for sentence in sentences])
    return np.amax(arr), np.amin(arr), np.average(arr), np.median(arr)


def generate_word2Vec(data, vector_len):
    tokenized_sentences = tokenize_sentences(data)
    model = Word2Vec(tokenized_sentences, size=vector_len)
    model.save("models/word2Vec_model")
    return model


def get_labeled_comments(df):
    mask = (df['toxic'] == 1) | (df['severe_toxic'] == 1) | (df['obscene'] == 1) | \
           (df['threat'] == 1) | (df['insult'] == 1) | (df['identity_hate'])
    return df[mask].drop('id', 1)


def load_vocab(word2vec_model):
    word2vec = Word2Vec.load(word2vec_model)
    return {k: v for k, v in zip(word2vec.wv.index2word, range(0, len(word2vec.wv.index2word)))}

def tokenize_comments(comments_list):
    tokenized_comments = []
    for comment in comments_list:
        tokens = word_tokenize(comment.strip())
        cleaned_tokens = []
        for token in tokens:
            cleaned_tokens.append(re.sub('[^A-Za-z0-9]+', '', token).strip().lower())
        tokenized_comments.append(list(filter(None, cleaned_tokens)))
    return tokenized_comments


def encode_comments(x_train, max_len, vocab):
    tokenized_comments = tokenize_comments(x_train)
    encoded_comments = []
    for comment in tokenized_comments:
        encoded_comment = [vocab[token] if token in vocab else None for token in comment]
        encoded_comments.append((list(filter(None, encoded_comment))))
    return pad_sequences(encoded_comments, maxlen=max_len, padding='post')


def build_multiclass_lstm_model(vocab_size, embedding_matrix, word_vector_lenth, avg_sentence_len):
    model = Sequential()
    model.add(Embedding(vocab_size, word_vector_lenth, weights=[embedding_matrix],
                        input_length=avg_sentence_len, trainable=False))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.25))
    model.add(Dense(6, activation='softmax'))
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_embedding_matrix(vocab_size, word_vec_len):
    word2vec = Word2Vec.load('models/word2Vec_model')
    w2v = dict(zip(word2vec.wv.index2word, word2vec.wv.syn0))
    embedding_matrix = np.zeros((vocab_size, word_vec_len))
    for word, i in vocab.items():
        embedding_vector = w2v.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


if __name__ == '__main__':

    word_vector_len = 256
    data = pd.read_csv('data/train.csv')
    word2Vec_model = None
    if os.path.exists('models/word2Vec_model'):
        word2Vec_model = Word2Vec.load('models/word2Vec_model')
    else:
        word2Vec_model = generate_word2Vec(data['comment_text'].tolist(), word_vector_len)

    train = get_labeled_comments(data)
    X_train, X_test, y_train, y_test = train_test_split(train['comment_text'], train[['toxic', 'severe_toxic',
                                                        'obscene', 'threat', 'insult', 'identity_hate']],
                                                        test_size=0.2, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    max, min, avg, median = get_sentence_len_stat(X_train)
    print ("Max : %s, Min = %s, Avg : %s, Median : %s" % (max, min, avg, median))

    vocab = load_vocab('models/word2Vec_model')
    vocab_size = len(vocab)

    X_train_encoded = encode_comments(X_train.tolist(), int(avg), vocab)
    X_val_encoded = encode_comments(X_val.tolist(), int(avg), vocab)
    embedding_matrix = get_embedding_matrix(vocab_size, word_vector_len)
    model = build_multiclass_lstm_model(vocab_size, embedding_matrix, word_vector_len, int(avg))
    model.fit(X_train_encoded, y_train , epochs=20)
    model.save("models/models_multi.h5")

    # evaluate the model
    comments = encode_comments(X_test.tolist(), int(avg), vocab)
    loss, accuracy = model.evaluate(comments, y_test, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))

    classes = model.predict(X_test)
    print (classes)

