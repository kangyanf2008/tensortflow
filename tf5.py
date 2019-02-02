import tensorflow as tf

#定义输入和参数 体积0.7,重量0.5
x = tf.constant([[0.7, 0.5]])
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

#定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y))
    #
    # x = tf.placeholder(tf.float32, shape=(None, 2))
    # init_op = tf.global_variables_initializer()
    # print(sess.run(y, feed_dict={x: [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]}))
