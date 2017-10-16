import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N, D, H = 64, 1000, 100

x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.placeholder(tf.float32, shape=(D, H))
w2 = tf.placeholder(tf.float32, shape=(H, D))

h = tf.maximum(0., tf.matmul(x, w1))
y_pred = tf.matmul(h, w2)

diff = y_pred - y
loss =  tf.reduce_mean(
    tf.reduce_mean(
        diff **2, axis=1,
    ),
)

grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

lr = 1e-3
losses = []
with tf.Session() as sess:
    values = {
        x: np.random.randn(*x.shape),
        y: np.random.randn(*y.shape),
        w1: np.random.randn(*w1.shape),
        w2: np.random.randn(*w2.shape),
    }

    for i in range(50):
        out = sess.run(
            [loss, grad_w1, grad_w2],
            feed_dict=values,
        )
        loss_val, grad_w1_val, grad_w2_val = out
        # print(loss_val)
        values[w1] -= lr * grad_w1_val
        values[w2] -= lr * grad_w2_val
        losses.append(loss_val)

plt.plot(losses)
plt.show()
