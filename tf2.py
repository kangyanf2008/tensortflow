import tensorflow as tf

a = tf.constant([[2.0, 3.0]], name="a")
b = tf.constant([[1.0], [4.0]], name="b")
result = tf.matmul(a, b, name="mul")
print(result)

with tf.Session() as sess:
    print(sess.run(result))

a = tf.constant([2.0, 3.0], name="a", shape=(2, 0), dtype="float64", verify_shape="false")
print(a)


var1 = tf.Variable([1, 3], name="v1")
var2 = tf.Variable([2, 4], name="v2")
init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "test/save.ckpt")
