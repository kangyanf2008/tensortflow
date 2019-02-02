import tensorflow as tf

w = tf.Variable(tf.random_normal([2, 3], stddev=2, mean=0, seed=1))

print(w)


