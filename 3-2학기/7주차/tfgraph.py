import tensorflow as tf
a = tf.random.uniform([2, 3], seed=3)
b = tf.random.normal([2, 3])
v = [[2, 3], [4, 5], [6, 7]]
c = tf.random.shuffle(v)

tf.print(a)
tf.print(b)
tf.print(c)
