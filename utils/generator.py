import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split


class ArithmeticGenerator:
    def __init__(self, output_root, ops, bound, test_size=0.1, seed=1234):
        self.ops = tuple(ops)
        self.bound = bound
        self.test_size = test_size
        self.rand = np.random.RandomState(seed)
        self.probSet = {}

        if not os.path.exists(output_root):
            os.mkdir(output_root)
        self.output_path = os.path.join(output_root, f'train_arith_{len(ops)}_{bound}_{seed}')

    def addProb(self, problem):
        self.probSet[problem] = str(eval(problem))

    def generate(self):
        for i in range(self.bound):
            for j in range(self.bound):
                for op in self.ops:
                    self.addProb(f'{i}{op}{j}')

    def save(self):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        df = pd.DataFrame({"Question": list(self.probSet.keys()),
                           "Answer": list(self.probSet.values())})
        print('Sample problems')
        print(df.head())
        print(f'saving files to {self.output_path}')
        df.to_csv(os.path.join(self.output_path, 'all.csv'), index=False)
        train, val = train_test_split(df, test_size=self.test_size, random_state=self.rand)
        train.to_csv(os.path.join(self.output_path, 'train.csv'), index=False)
        val.to_csv(os.path.join(self.output_path, 'val.csv'), index=False)


class MixedArithmeticGenerator(ArithmeticGenerator):
    def __init__(self, output_root, ops, bound, test_size=0.1, seed=1234):
        super().__init__(output_root, ops, bound, test_size, seed)
        self.output_path = os.path.join(output_root, f'train_mixedarith_{len(ops)}_{bound}_{seed}')

    def generate(self):
        for i in range(self.bound):
            for j in range(self.bound):
                for op in self.ops:
                    self.addProb(f'{i}{op}{j}')
                    rand_op = self.rand.choice(self.ops)
                    rand_num = self.rand.randint(self.bound)
                    self.addProb(f'{i}{op}{j}{rand_op}{rand_num}')


# class MathLang:
#     def __init__(self, name):
#         self.name = name
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {0: "<sos>", 1: "<eos>"}
#         self.n_words = 2  # Count SOS and EOS
#
#     def addSentence(self, sentence):
#         for word in sentence:
#             self.addWord(word)
#
#     def addWord(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.n_words
#             self.word2count[word] = 1
#             self.index2word[self.n_words] = word
#             self.n_words += 1
#         else:
#             self.word2count[word] += 1
