# import tensorflow as tf
m1 = tf.constant([3, 5])
m2 = tf.constant([2, 4])
result = tf.add(m1, m2)
print(result)

sess = tf.Session()
print(sess.run(result))
sess.close()

with tf.Session() as sess:
    res = sess.run([result])
print(res)

with tf.Session() as sess:
    with tf.device("/gpu:2"):
        m1 = tf.constant([3, 5])
        m2 = tf.constant([2, 4])
        result = tf.add(m1, m2)
print(result)

m1 = tf.constant([3, 5])
m2 = tf.constant([2, 4])
result = tf.add(m1, m2)
sess = tf.InteractiveSession()
print(result.eval())