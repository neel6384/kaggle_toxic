import re
import os.path
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


def is_foul(train_raw):
    if train_raw['toxic'] == 1 or train_raw['severe_toxic'] == 1 or train_raw['obscene'] == 1 or\
                    train_raw['threat'] == 1 or train_raw['insult'] == 1 or train_raw['identity_hate']:
        return 1
    else:
        return 0


def get_tokenized_sentences(data):
    tokenized_sentences = []
    for comment in data:
        sentences = sent_tokenize(comment.decode('utf-8').strip())
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            cleaned_tokes = []
            for token in tokens:
                cleaned_tokes.append(re.sub('[^A-Za-z0-9]+', '', token).strip().lower())
            tokenized_sentences.append(list(filter(None, cleaned_tokes)))
    return tokenized_sentences


def comments_to_token(data):
    tokenized_comments = []
    for comment in data:
        tokens = word_tokenize(comment.decode('utf-8').strip())
        cleaned_tokens = []
        for token in tokens:
            cleaned_tokens.append(re.sub('[^A-Za-z0-9]+', '', token).strip().lower())
        tokenized_comments.append(list(filter(None, cleaned_tokens)))

    return tokenized_comments


def generate_word2vec(tokenized_sentences):
    model = Word2Vec(tokenized_sentences)
    model.save('models/word2Vec_model')


def get_word2vec(tokenized_sentences):
    if os.path.exists('models/word2Vec_model'):
        return Word2Vec.load('models/word2Vec_model')
    else:
        generate_word2vec(tokenized_sentences)
        return Word2Vec.load('models/word2Vec_model')


def get_encoded_comments(tokenized_comments, vocab):
    encoded_comments = []
    for comment in tokenized_comments:
        encoded_comment = [vocab[token] if token in vocab else None for token in comment]
        encoded_comments.append((list(filter(None, encoded_comment))))
    return encoded_comments