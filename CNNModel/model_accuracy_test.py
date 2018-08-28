import tensorflow as tf
from file_reader import Reader
from convolutional_nn import CNN
from meta import Meta

if __name__ == '__main__':
    test_path = 'Input/test.tfrecords'
    batch_size = 100
    meta_dict = Meta().load_dict('Input/batches_meta.json')
    num_examples = meta_dict['num_examples']['test']
    num_batches = int(num_examples / batch_size)

    print('===> EVALUATING LAST MODEL ON TEST SAMPLES .....')

    with tf.Graph().as_default():

        images, labels = Reader.build_batch(test_path, batch_size=batch_size,
                                            file_length=num_examples, shuffled=False)
        logits = CNN.model(images)
        predictions = tf.argmax(logits, axis=1)

        accuracy, update_accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            restorer = tf.train.Saver()
            checkpoint_path = tf.train.latest_checkpoint('logs/train')
            restorer.restore(sess, checkpoint_path)

            for _ in range(num_batches):
                sess.run(update_accuracy)

            predictions_test, accuracy_test = sess.run([predictions, accuracy])

            coord.request_stop()
            coord.join(threads)

    print('The model accuracy on test samples is %f:' % accuracy_test)
