## Dependencies

* Python 3.5 or +
* tensorflow 1.8
* tensorboard 1.8

## How to use

#### Download and create input file in .tfrecords

If you have already downloaded the \*tar.gz file create a directory *Input* inside the directory *CNNModel* and put the .tar file inside it.   
Then call the script file_creator, it will extract and transform the files.

Otherwise calling the script should automatically
download the file, but this operation could fail for certain OS.

```
    $ python file_creator.py
```


#### Train Model
Move inside *CNNModel* directory and run

```
    $ python train.py
```

If a model already exists and you want to train from there:

```
    $ python train.py --path path_to_model/model_name.ckpt
```

If you want to restart from the last checkpoint

```
 $ python train.py --last True
```

#### Evaluate Model on Test Samples
Move inside *CNNModel* directory and run

```
    $ python model_accuracy_test.py
```

#### Visualize training on tensorboard
Move inside *CNNModel* directory and run


```
    $ tensorboard --logdir ./logs
```

