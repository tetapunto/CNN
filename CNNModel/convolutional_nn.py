import tensorflow as tf


class CNN:

    @staticmethod
    def model(x):
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[4, 4],
            padding='SAME',
            activation=tf.nn.relu
        )

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )

        pool1 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, padding='SAME')
        drop = tf.layers.dropout(pool1, rate=0.25)

        conv3 = tf.layers.conv2d(
            inputs=drop,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )

        pool2 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2, padding='SAME')

        conv4 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )

        conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=128,
            kernel_size=[2, 2],
            padding='SAME',
            activation=tf.nn.relu
        )

        pool3 = tf.layers.max_pooling2d(conv5, pool_size=[2, 2], strides=2, padding='SAME')
        batch_size = x.get_shape()[0].value
        flat = tf.reshape(pool3, [batch_size, 4 * 4 * 128])

        dense = tf.layers.dense(inputs=flat, units=1500, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=10)
        return logits

    @staticmethod
    def loss(logits, labels):
        cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))
        loss = tf.reduce_sum(cross_entropy)
        return loss
