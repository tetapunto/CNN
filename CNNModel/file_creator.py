import os
import pickle
import tensorflow as tf
from meta import Meta
import tarfile
import urllib
import shutil
import argparse


class InputCreator:
    download_name = 'cifar-10-python.tar.gz'
    download_url = 'https://www.cs.toronto.edu/~kriz/' + download_name
    _batch_images = None
    _batch_labels = None
    _num_examples = None
    sample_index = None
    input_dir = './Input'

    def __init__(self):
        InputCreator.check_input_dir(self.input_dir)
        InputCreator.downloader(self.input_dir)

    def set_example_reader(self, batch_filename):
        with open(self.input_dir + '/cifar-10-batches-py/' + batch_filename, 'rb') as f:
            data_batch = pickle.load(f, encoding='latin1')
        self._batch_images = data_batch['data']
        self._batch_labels = data_batch['labels']
        self._num_examples = self._batch_images.shape[0]
        self.sample_index = 0

    def _single_sample_reader(self):
        if self.sample_index == self._num_examples or self.sample_index is None:
            return None
        image = self._batch_images[self.sample_index].reshape([3, 32, 32]).transpose(1, 2, 0).tostring()
        label = self._batch_labels[self.sample_index]
        self.sample_index += 1
        example = tf.train.Example(features=tf.train.Features(
            feature={'image': InputCreator._bytes_feature(image), 'label': InputCreator._int64_feature(label)}))
        return example

    def convert_and_write(self):
        dict_sample_number = {}
        writers = {}

        for saved_file_name in ['train', 'val', 'test']:
            dict_sample_number[saved_file_name] = 0
            writers[saved_file_name] = tf.python_io.TFRecordWriter(
                self.input_dir + '/' + saved_file_name + '.tfrecords')

        for file in os.listdir(self.input_dir + '/cifar-10-batches-py'):
            key = self.check_train_val_test(file)
            if key is not None:
                print('Processing {} as {}...'.format(file, key))
                self.set_example_reader(file)
                while True:
                    example = self._single_sample_reader()
                    if example is None:
                        break

                    writers[key].write(example.SerializeToString())
                    dict_sample_number[key] += 1
            else:
                continue

        for writer in writers.values():
            writer.close()

        return dict_sample_number

    def create_tfrecords_meta_file(self, dict_sample_number, meta_name='batches_meta.json'):
        print('Saving meta file to %s...' % self.input_dir)
        meta = Meta(**dict_sample_number)
        with open(self.input_dir + '/cifar-10-batches-py/batches.meta', 'rb') as f:
            content = pickle.load(f, encoding='latin1')
            meta.categories = content['label_names']
        meta.save(self.input_dir + '/' + meta_name)

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def check_input_dir(input_dir):
        if not os.path.isdir(input_dir):
            os.mkdir(input_dir)

    @classmethod
    def downloader(cls, input_dir):
        if not os.path.isfile(input_dir + '/' + cls.download_name):
            print('Downloading from {}'.format(cls.download_url))
            with urllib.request.urlopen(cls.download_url) as response, open(cls.download_name, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            os.rename(cls.download_name, input_dir + '/' + cls.download_name)
        else:
            print('{} already downloaded'.format(cls.download_name))

        if not os.path.isdir(input_dir + '/cifar-10-batches-py'):
            print('Extracting file...')
            tarfile.open(os.path.join(input_dir, 'cifar-10-python.tar.gz'), 'r:gz').extractall(input_dir)

    @staticmethod
    def check_train_val_test(filename):
        if filename in ['data_batch_' + str(i) for i in range(1, 5)]:
            key = 'train'
        elif filename == 'data_batch_5':
            key = 'val'
        elif filename == 'test_batch':
            key = 'test'
        else:
            key = None
        return key


def main(_):
    creator = InputCreator()
    dict_sample_numbers = creator.convert_and_write()
    creator.create_tfrecords_meta_file(dict_sample_numbers)
    print('Done!')


if __name__ == '__main__':
    tf.app.run(main=main)
