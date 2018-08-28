import os
from datetime import datetime
import time
import tensorflow as tf
from meta import Meta
from file_reader import Reader
from evaluator import Evaluator
from file_creator import InputCreator
from convolutional_nn import CNN

import argparse


class Trainer:
    input_dir = InputCreator.input_dir
    training_set = os.path.join(input_dir, 'train.tfrecords')
    val_set = os.path.join(input_dir, 'val.tfrecords')
    test_set = os.path.join(input_dir, 'test.tfrecords')
    num_examples = Meta.load_dict(os.path.join(input_dir, 'batches_meta.json'))['num_examples']
    categories = Meta.load_dict(os.path.join(input_dir, 'batches_meta.json'))['categories']
    log_train_dir = './logs/train'

    def __init__(self, start_from_specific_checkpoint=None, start_from_last_checkpoint=False):

        if start_from_specific_checkpoint is not None and start_from_last_checkpoint:
            raise ImportError('You can specify a single checkpoint at time!!')

        self.start_from_specific_checkpoint = start_from_specific_checkpoint
        self.start_from_last_checkpoint = start_from_last_checkpoint

    def train(self, batch_size=128, initial_patience=200, iter_check_loss=10):

        with tf.Graph().as_default():
            images, labels = Reader.build_batch(Trainer.training_set, file_length=Trainer.num_examples['train'],
                                                batch_size=batch_size, shuffled=True)
            logits = CNN.model(images)
            loss = CNN.loss(logits, labels)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf.train.exponential_decay(0.1, global_step=global_step, decay_steps=10000, decay_rate=0.01)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step)

            predictions = tf.argmax(logits, axis=1)
            training_accuracy, up_training_accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)

            tf.summary.scalar('training_accuracy', training_accuracy)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('learning_rate', learning_rate)
            summary = tf.summary.merge_all()

            with tf.Session() as sess:
                summary_writer = tf.summary.FileWriter(Trainer.log_train_dir, sess.graph)
                evaluator = Evaluator(os.path.join(Trainer.log_train_dir, 'eval/val'))

                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

                sess.run(init_op)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                saver = tf.train.Saver()
                if self.start_from_specific_checkpoint is not None:
                    saver.restore(sess, self.start_from_specific_checkpoint)
                    print('Model restored from file: %s' % self.start_from_specific_checkpoint)

                if self.start_from_last_checkpoint:
                    checkpoint_path = tf.train.latest_checkpoint('logs/train')
                    saver.restore(sess, checkpoint_path)
                    print('Model restored from last checkpoint')

                with open('./Utils/start.txt') as f:
                    print(f.read())
                patience = initial_patience
                best_accuracy = 0.0
                duration = 0.0

                while True:
                    start_time = time.time()
                    _, loss_val, accuracy_train, summary_val, global_step_val = sess.run(
                        [train_op, loss, training_accuracy, summary, global_step])
                    duration += time.time() - start_time

                    if global_step_val % iter_check_loss == 0:
                        examples_per_sec = batch_size * iter_check_loss / duration
                        duration = 0.0
                        print('=> %s: step %d, loss = %f (%.1f examples/sec)' % (
                            datetime.now(), global_step_val, loss_val, examples_per_sec))

                    summary_writer.add_summary(summary_val, global_step=global_step_val)

                    print('---------> Evaluating on validation dataset... <---------')
                    path_to_latest_checkpoint_file = saver.save(sess,
                                                                os.path.join(Trainer.log_train_dir, 'latest.ckpt'))
                    accuracy = evaluator.evaluate(path_to_latest_checkpoint_file, Trainer.val_set,
                                                  Trainer.num_examples['val'],
                                                  global_step_val)

                    print('=---------> accuracy_validation = %f, best accuracy %f  <---------' % (
                        accuracy, best_accuracy))

                    if accuracy > best_accuracy:
                        path_to_checkpoint_file = saver.save(sess,
                                                             os.path.join(Trainer.log_train_dir, 'model.ckpt'),
                                                             global_step=global_step_val)
                        print('=> Model saved to file: %s' % path_to_checkpoint_file)
                        patience = initial_patience
                        best_accuracy = accuracy
                    else:
                        patience -= 1

                    print('=> patience = %d' % patience)
                    if patience == 0:
                        break

                coord.request_stop()
                coord.join(threads)
                with open('./Utils/done.txt') as f:
                    print(f.read())


def main(args):
    Trainer(start_from_specific_checkpoint=args.path,
            start_from_last_checkpoint=args.last).train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', action='store', type=str, help='Path of the model you want to start from',
                        default=None)

    parser.add_argument('--last', action='store', type=bool, help='If true training start from last valid checkpoint',
                        default=False)
    main(parser.parse_args())
