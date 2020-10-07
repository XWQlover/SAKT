import pandas as pd
import tensorflow as tf
import numpy as np

class Assisment():
    """
    format
    15,16,17,18,-1,-1,-1,-1
    0,1,1,1,-1,-1,-1,-1
    """

    def __init__(self):
        self.data = pd.read_csv("C:\\Users\\Administrator\\Desktop\\data\\2015_100_skill_builders_main_problems.csv")
        self.data = self.data.drop("log_id", axis=1)
        self.data.dropna()
        self.data["user_id"], _ = pd.factorize(self.data["user_id"])
        self.data["sequence_id"], _ = pd.factorize(self.data["sequence_id"])
        self.data["skills_correctness"] = self.data.apply(
            lambda x: x.sequence_id * 2 if x.correct == 0.0 else x.sequence_id * 2 + 1, axis=1)
        self.data = self.data.groupby("user_id").filter(lambda q: len(q) > 1).copy()
        self.seqLen = max(self.data.groupby('user_id').apply(lambda r: len(r["sequence_id"].values)))
        self.seq = self.data.groupby('user_id').apply(
            lambda r: (
                r["skills_correctness"].values[:-1],
                r["sequence_id"].values[1:],
                r['correct'].values[1:]
            )
        )
        self.df = pd.DataFrame()
        pad = [-1] * self.seqLen
        for seq in self.seq:
            l = []
            for i in range(len(seq[1])):
                l.append((seq[0][:i + 1].tolist() + pad[:self.seqLen - i], [seq[1][i]], [seq[2][i]]))
            self.df = self.df.append(l)
        self.df = self.df.sample(frac=1)

    def datasetReturn(self, shuffle=None, batch_size=32, val_data=None):
        df = self.df

        def generator():
            nonlocal df
            for index, i in df.iterrows():
                yield i[0], i[1], i[2]

        dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.int32, tf.int32, tf.int32))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle)
        i = 0
        for l in dataset.as_numpy_iterator():
            i += 1
        print(i)
        # split
        test_size = int(np.ceil(i * 0.2))
        train_size = i - test_size

        train_data = dataset.take(train_size)
        dataset = dataset.skip(train_size)

        return train_data, dataset