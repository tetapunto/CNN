import json


class Meta(object):
    def __init__(self, categories=None, train=None, val=None, test=None):
        self.len_train = train
        self.len_val = val
        self.len_test = test
        self.categories = categories

    def save(self, path_to_json_file):
        with open(path_to_json_file, 'w') as f:
            content = {
                'num_examples': {
                    'train': self.len_train,
                    'val': self.len_val,
                    'test': self.len_test
                },
                'categories': self.categories
            }
            json.dump(content, f)

    @staticmethod
    def load_dict(path_to_json_file):
        with open(path_to_json_file, 'r') as f:
            content = json.load(f)
            return content
