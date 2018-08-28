import tensorflow as tf


class Reader:
    @staticmethod
    def _preprocess(image):
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image = tf.reshape(image, [32, 32, 3])
        return image

    @staticmethod
    def _read_and_decode(filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        image = Reader._preprocess(image)
        label = tf.cast(features['label'], tf.int32)

        return image, label

    @staticmethod
    def build_batch(file_path, file_length, batch_size, shuffled):

        filename_queue = tf.train.string_input_producer([file_path], num_epochs=None)
        image, label = Reader._read_and_decode(filename_queue)

        min_queue_examples = int(0.4 * file_length)
        if shuffled:
            images, labels = tf.train.shuffle_batch([image, label],
                                                    batch_size=batch_size,
                                                    num_threads=2,
                                                    capacity=min_queue_examples + 3 * batch_size,
                                                    min_after_dequeue=min_queue_examples)
        else:
            images, labels = tf.train.batch([image, label],
                                            batch_size=batch_size,
                                            num_threads=2,
                                            capacity=min_queue_examples + 3 * batch_size)
        return images, labels
