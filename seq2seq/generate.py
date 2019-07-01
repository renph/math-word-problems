import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split


class ArithmeticGenerator:
    def __init__(self, output_path, ops=('+', '-'), test_size=0.1, seed=1234):
        self.ops = tuple(ops)
        self.output_path = output_path
        self.test_size = test_size
        self.rand = np.random.RandomState(seed)

        self.probSet = {}

    def addProb(self, problem):
        self.probSet[problem] = str(eval(problem))

    def generate(self, bound=10):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        for i in range(bound):
            for j in range(bound):
                for op in self.ops:
                    self.addProb(f'{i}{op}{j}')
        df = pd.DataFrame({"Question": list(self.probSet.keys()),
                           "Answer": list(self.probSet.values())})
        print(df.head())
        df.to_csv(os.path.join(self.output_path, 'all.csv'), index=False)
        train, val = train_test_split(df, test_size=self.test_size, random_state=self.rand)
        train.to_csv(os.path.join(self.output_path, 'train.csv'), index=False)
        val.to_csv(os.path.join(self.output_path, 'val.csv'), index=False)


class MixedArithmeticGenerator(ArithmeticGenerator):
    def generate(self, bound=10):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        for i in range(bound):
            for j in range(bound):
                for op in self.ops:
                    self.addProb(f'{i}{op}{j}')
                    rand_op = self.rand.choice(self.ops)
                    rand_num = self.rand.randint(bound)
                    self.addProb(f'{i}{op}{j}{rand_op}{rand_num}')

        df = pd.DataFrame({"Question": list(self.probSet.keys()),
                           "Answer": list(self.probSet.values())})
        print(df.head())
        df.to_csv(os.path.join(self.output_path, 'all.csv'), index=False)
        train, val = train_test_split(df, test_size=self.test_size, random_state=self.rand)
        train.to_csv(os.path.join(self.output_path, 'train.csv'), index=False)
        val.to_csv(os.path.join(self.output_path, 'val.csv'), index=False)


if __name__ == '__main__':
    bound = 100
    seed = 4111
    ops = ('+', '-', '*')
    gen = ArithmeticGenerator(f'../tmp/train_arith_{len(ops)}_{bound}_{seed}', ops=ops, seed=seed,
                              test_size=0.8)
    gen.generate(bound)

    # bound = 100
    # seed = 1111
    # ops = ('+', '-', '*')
    # gen = MixedArithmeticGenerator(f'../tmp/train_mixedarith_{len(ops)}_{bound}_{seed}', ops=ops, seed=seed)
    # gen.generate(bound)
