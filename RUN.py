import pandas as pd
import tensorflow as tf
import numpy as np
from ASS2015 import Assisment
from SAKT import MultiHeadAttention

ass = Assisment()
mask = -1
train_data, test_data = ass.datasetReturn()
train_data, test_data = train_data.batch(100), test_data.batch(100)


total_skills_correctness = max(ass.data["skills_correctness"].unique())+1
sakt = MultiHeadAttention(int(total_skills_correctness),512,8)
vauc = tf.metrics.AUC()
bc = tf.metrics.BinaryAccuracy()
auc = tf.metrics.AUC()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# summary_writer = tf.summary.create_file_writer(val_log)


def test_one_step(q, k, v, correctness):
    probility = sakt(q, k, v)  # (batch , 2)

    vauc.update_state(correctness, probility)


def train_one_step(q, k, v, correctness):
    with tf.GradientTape() as tape:
        probility = sakt(q, k, v)  # (batch , 2)

        bc.update_state(correctness, probility)

        loss = tf.keras.losses.binary_crossentropy(correctness, probility)
        loss = tf.reduce_sum(loss)

        auc.update_state(correctness, probility)

        gradients = tape.gradient(loss, sakt.trainable_variables)
        # 反向传播，自动微分计算
        optimizer.apply_gradients(zip(gradients, sakt.trainable_variables))


import time

for epoch in range(6):
    start = time.time()
    # train_data = train_data.shuffle(32)
    auc.reset_states()
    vauc.reset_states()
    bc.reset_states()
    for k, q, c in train_data.as_numpy_iterator():
        train_one_step(q, k, k, c)
        # print(attention(q1,k1,k1))

    for k, q, c in test_data.as_numpy_iterator():
        test_one_step(q, k, k, c)

    # with summary_writer.as_default():
    #     tf.summary.scalar('train_auc', auc.result(), step=epoch)
    #     tf.summary.scalar('val_auc', vauc.result(), step=epoch)

    print("time:{0}".format(time.time() - start), cc.result(), auc.result(), vauc.result())