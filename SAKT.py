import pandas as pd
import tensorflow as tf
import numpy as np
batch_size = 100
embeddingsize = 32
@tf.function
def scaled_dot_product_attention(q, k, v, mask, num_heads):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_weights = tf.nn.softmax(scaled_attention_logits,
                                      axis=-1)  # (..., seq_len_q, seq_len_k) (batchsize,heads,1,seqlen)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, total_skills_correctness, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.mask = tf.keras.layers.Masking(mask_value=-1)
        self.skill_correct_embedding = tf.keras.layers.Embedding(total_skills_correctness, embeddingsize)
        self.skill_embedding = tf.keras.layers.Embedding(int(total_skills_correctness / 2), embeddingsize)
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, activation='sigmoid')
        self.wk = tf.keras.layers.Dense(d_model, activation='sigmoid')
        self.wv = tf.keras.layers.Dense(d_model, activation='sigmoid')

        self.dense = tf.keras.layers.Dense(d_model, activation='relu')
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model / num_heads, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(1, activation='sigmoid')  # (batch_size, seq_len, d_model)
        ])

    def call(self, q, k, v):
        """
        :param v:  和k同
        :param k:  (batch ,seqlen,1)的序列
        :param q:  ( 1,)
        :return:
        """
        k = tf.expand_dims(k, axis=-1)
        v = tf.expand_dims(v, axis=-1)
        mask = 1. - tf.cast(tf.equal(k, -1), tf.float32)  # (batch , seq_len ,)

        k = self.mask(k)
        v = self.mask(v)
        k = self.skill_correct_embedding(k)  # (batch , seq_len, d)
        q = self.skill_embedding(q)  # (batch , d)
        v = self.skill_correct_embedding(v)
        v = tf.squeeze(v, axis=2)  # (batch , seq_len, d)

        q = tf.reshape(q, (-1, 1, 32))
        k = tf.reshape(k, (-1, 633, 32))

        k = k * mask
        v = v * mask
        # 正确
        q_ = self.wq(q)  # (batch_size, 1,d_model)
        k = self.wk(k) * mask  # (batch_size, seq_len, d_model)
        v = self.wv(v) * mask  # (batch_size, seq_len, d_model)

        q_ = tf.transpose(tf.reshape(q_, (-1, 1, self.num_heads, self.depth)),
                          perm=[0, 2, 1, 3])  # (batchsize , heads ,seqlen ,depth)
        k = tf.transpose(tf.reshape(k, (-1, 633, self.num_heads, self.depth)), perm=[0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, (-1, 633, self.num_heads, self.depth)), perm=[0, 2, 1, 3])

        scaled_attention, attention_weights = scaled_dot_product_attention(q_, k, v, mask, self.num_heads)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (-1, self.d_model))  # (batch_size, seq_len_q, d_model)
        # print(concat_attention)
        # concat_attention = tf.reduce_sum(concat_attention,axis=1) # (batch_size, d_model)
        concat_attention = tf.concat([concat_attention, tf.reshape(q, (-1, 32))], axis=-1)

        output = self.ffn(self.dense(concat_attention))  # (batch_size, seq_len_q, d_model)

        return output