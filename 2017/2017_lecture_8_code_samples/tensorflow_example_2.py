import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

N, D, H = 64, 1000, 100

x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.Variable(tf.random_normal((D, H)))
w2 = tf.Variable(tf.random_normal((H, D)))

h = tf.maximum(0., tf.matmul(x, w1))
y_pred = tf.matmul(h, w2)

diff = y_pred - y
loss =  tf.reduce_mean(
    tf.reduce_mean(
        diff **2, axis=1,
    ),
)

grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
losses = []

lr = 1e-3
new_w1 = w1.assign(w1 - lr * grad_w1)
new_w2 = w2.assign(w2 - lr * grad_w2)

# dummy tensor to make the graph run this calculation
updates =  tf.group(new_w1, new_w2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {
        x: np.random.randn(*x.shape),
        y: np.random.randn(*y.shape),
    }

    for i in range(50):
        loss_val, _ = sess.run(
            [loss, updates],
            feed_dict=values,
        )
        # loss_val, grad_w1_val, grad_w2_val = out
        print(loss_val)
        # losses.append(loss_val)
