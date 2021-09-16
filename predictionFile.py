# load and evaluate a saved model
import numpy as np
from keras.models import load_model
from utils import vectorize_stories
import tarfile
from keras.utils.data_utils import get_file
from functools import reduce
import re
from keras.layers import Input
from keras.preprocessing.sequence import pad_sequences


class Prediction:

    def __init__(self):
        # load trained model
        self.model = load_model("model.h5", compile=False)
        # model.summary()

        self.story_maxlen = 68
        self.query_maxlen = 4

    def loadDataset(self):
        try:
            path = get_file("dataset","http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz")
        except:
            print('Error downloading dataset, please download it manually:\n'
                  '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
                  '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
            raise

        tar = tarfile.open(path)
        return tar

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

    def getTrainTestDataset(self, vocab):

        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        idx_word = dict((i + 1, c) for i, c in enumerate(vocab))

        return word_idx, idx_word

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

        return vocab

    def getPrediction(self, user_story, user_query, word_idx, idx_word):
        user_story_vectorise, user_query_vectorise, user_ans = vectorize_stories([[user_story, user_query, '.']], word_idx,
                                                             self.story_maxlen, self.query_maxlen)
        user_prediction = self.model.predict([user_story_vectorise, user_query_vectorise])
        user_prediction = idx_word[np.argmax(user_prediction)]

        return user_prediction

    def executeProcessing(self, user_story, user_query):
        tar = self.loadDataset()
        vocab = self.parseDataset(tar)
        word_idx, idx_word = self.getTrainTestDataset(vocab)

        prediction = self.getPrediction(user_story, user_query, word_idx, idx_word)
        return prediction
