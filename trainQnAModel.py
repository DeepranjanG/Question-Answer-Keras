"""
Trains a memory network on the facebook bAbI dataset for Question/Answering System.
"""

# import the packages
import keras
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM, GRU
from keras.utils.data_utils import get_file
from keras import backend as K

from functools import reduce
import tarfile
import numpy as np
import re

from utils import vectorize_stories


class ModelTraining:

    def __init__(self):
        self.train_epochs = 100
        self.batch_size = 32
        self.lstm_size = 64

    def tokenize(self, sent):
        '''Return the tokens of a sentence including punctuation.
        tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        '''
        return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

    def parse_stories(self, lines, only_supporting=False):
        '''Parse stories provided in the bAbi tasks format
        If only_supporting is true, only the sentences
        that support the answer are kept.
        '''
        data = []
        story = []
        for line in lines:
            line = line.decode('utf-8').strip()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = self.tokenize(q)
                substory = None
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = self.tokenize(line)
                story.append(sent)
        return data

    def get_stories(self, f, only_supporting=False, max_length=None):
        """Given a file name, read the file,
        retrieve the stories,
        and then convert the sentences into a single story.
        If max_length is supplied,
        any stories longer than max_length tokens will be discarded.
        """
        data = self.parse_stories(f.readlines(), only_supporting=only_supporting)
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data if
                not max_length or len(flatten(story)) < max_length]
        return data

    def downloadDataset(self):
        try:
            path = get_file('babi-tasks-v1-2.tar.gz',
                            origin='dataset')
        except:
            print('Error downloading dataset, please download it manually:\n'
                  '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
                  '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
            raise

        tar = tarfile.open(path)
        return tar

    def parseDataset(self, tar):
        challenges = {
            # QA1 with 10,000 samples
            'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
            # QA2 with 10,000 samples
            'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
        }
        challenge_type = 'single_supporting_fact_10k'
        challenge = challenges[challenge_type]

        print('Extracting stories for the challenge:', challenge_type)
        train_stories = self.get_stories(tar.extractfile(challenge.format('train')))
        test_stories = self.get_stories(tar.extractfile(challenge.format('test')))

        vocab = set()
        for story, q, answer in train_stories + test_stories:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        vocab_size = len(vocab) + 1
        story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
        query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

        print('-')
        print('Vocab size:', vocab_size, 'unique words')
        print('Story max length:', story_maxlen, 'words')
        print('Query max length:', query_maxlen, 'words')
        print('Number of training stories:', len(train_stories))
        print('Number of test stories:', len(test_stories))
        print('-')
        print('Here\'s what a "story" tuple looks like (input, query, answer):')
        print(train_stories[0])
        print('-')
        print('Vectorizing the word sequences...')

        return vocab, vocab_size, train_stories, test_stories, story_maxlen, query_maxlen

    def getTrainTestDataset(self, vocab, train_stories, test_stories, story_maxlen, query_maxlen):

        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        idx_word = dict((i + 1, c) for i, c in enumerate(vocab))
        inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                                       word_idx,
                                                                       story_maxlen,
                                                                       query_maxlen)
        inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                                    word_idx,
                                                                    story_maxlen,
                                                                    query_maxlen)

        print('-')
        print('inputs: integer tensor of shape (samples, max_length)')
        print('inputs_train shape:', inputs_train.shape)
        print('inputs_test shape:', inputs_test.shape)
        print('-')
        print('queries: integer tensor of shape (samples, max_length)')
        print('queries_train shape:', queries_train.shape)
        print('queries_test shape:', queries_test.shape)
        print('-')
        print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
        print('answers_train shape:', answers_train.shape)
        print('answers_test shape:', answers_test.shape)
        print('-')
        print('Compiling...')

        # placeholders
        input_sequence = Input((story_maxlen,))
        question = Input((query_maxlen,))

        print('Input sequence:', input_sequence)
        print('Question:', question)

        return word_idx, idx_word, inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test, input_sequence, question

    def buildModelArch(self, vocab_size, query_maxlen, input_sequence, question):
        # encoders
        # embed the input sequence into a sequence of vectors
        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=vocab_size,
                                      output_dim=64))
        input_encoder_m.add(Dropout(0.3))
        # output: (samples, story_maxlen, embedding_dim)

        # embed the input into a sequence of vectors of size query_maxlen
        input_encoder_c = Sequential()
        input_encoder_c.add(Embedding(input_dim=vocab_size,
                                      output_dim=query_maxlen))
        input_encoder_c.add(Dropout(0.3))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the question into a sequence of vectors
        question_encoder = Sequential()
        question_encoder.add(Embedding(input_dim=vocab_size,
                                       output_dim=64,
                                       input_length=query_maxlen))
        question_encoder.add(Dropout(0.3))
        # output: (samples, query_maxlen, embedding_dim)

        # encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        print('Input encoded m', input_encoded_m)
        input_encoded_c = input_encoder_c(input_sequence)
        print('Input encoded c', input_encoded_c)
        question_encoded = question_encoder(question)
        print('Question encoded', question_encoded)

        # compute a 'match' between the first input vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        print(match.shape)
        match = Activation('softmax')(match)
        print('Match shape', match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)
        print('Response shape', response)

        # concatenate the response vector with the question vector sequence
        answer = concatenate([response, question_encoded])
        print('Answer shape', answer)

        # answer = LSTM(lstm_size, return_sequences=True)(answer)  # Generate tensors of shape 32
        # answer = Dropout(0.3)(answer)
        answer = LSTM(self.lstm_size)(answer)  # Generate tensors of shape 32
        answer = Dropout(0.3)(answer)
        answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = Activation('softmax')(answer)

        # build the final model
        model = Model([input_sequence, question], answer)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("-------------Model Summary------------")
        print(model.summary())

        return model

    def trainModel(self, model, inputs_train, queries_train, answers_train, inputs_test,
                   queries_test, answers_test):
        #
        print("Trainig the model")
        model.fit([inputs_train, queries_train], answers_train, self.batch_size, self.train_epochs,
                  validation_data=([inputs_test, queries_test], answers_test))
        model.save('model.h5')

    def modelEvaluation(self, model, test_stories, word_idx, story_maxlen, query_maxlen, idx_word):
        print('-------------------------------------------------------------------------------------------')
        print('Qualitative Test Result Analysis')
        for i in range(0, 10):
            current_inp = test_stories[i]
            current_story, current_query, current_answer = vectorize_stories([current_inp], word_idx, story_maxlen,
                                                                                  query_maxlen)
            current_prediction = model.predict([current_story, current_query])
            current_prediction = idx_word[np.argmax(current_prediction)]
            print(' '.join(current_inp[0]), ' '.join(current_inp[1]), '| Prediction:', current_prediction,
                  '| Ground Truth:',
                  current_inp[2])

    def executeProcessing(self):
        tar = self.downloadDataset()
        vocab, vocab_size, train_stories, test_stories, story_maxlen, query_maxlen = self.parseDataset(tar)
        word_idx, idx_word, inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test, input_sequence, question = self.getTrainTestDataset(
            vocab, train_stories, test_stories, story_maxlen, query_maxlen)
        model = self.buildModelArch(vocab_size, query_maxlen, input_sequence, question)
        self.trainModel(model, inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test)
        self.modelEvaluation(model, test_stories, word_idx, story_maxlen, query_maxlen, idx_word)


if __name__ == "__main__":
    mdlTrnng = ModelTraining()
    mdlTrnng.executeProcessing()
