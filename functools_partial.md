In a nutshell, the partial() is used to “freeze” some portion of a function’s arguments and/or keywords resulting in a new object 
with a simplified signature. This possibly removes a lot of clutter where the same function is called many times with little parameters’ 
variation.

 Let’s take the definition of the first three blocks of layers of VGG19:

```python
conv1_1 = tf.layers.conv2d(image, filters=64, kernel_size=(3, 3), padding='same',
                           activation=tf.nn.relu, use_bias=True, name='conv1_1')
conv1_2 = tf.layers.conv2d(conv1_1, filters=64, kernel_size=(3, 3), padding='same',
                           activation=tf.nn.relu, use_bias=True, name='conv1_2')
pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')

conv2_1 = tf.layers.conv2d(pool1, filters=128, kernel_size=(3, 3), padding='same',
                           activation=tf.nn.relu, use_bias=True, name='conv2_1')
conv2_2 = tf.layers.conv2d(conv2_1, filters=128, kernel_size=(3, 3), padding='same',
                           activation=tf.nn.relu, use_bias=True, name='conv2_2')
pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')

conv3_1 = tf.layers.conv2d(pool2, filters=256, kernel_size=(3, 3), padding='same',
                           activation=tf.nn.relu, use_bias=True, name='conv3_1')
conv3_2 = tf.layers.conv2d(conv3_1, filters=256, kernel_size=(3, 3), padding='same',
                           activation=tf.nn.relu, use_bias=True, name='conv3_2')
conv3_3 = tf.layers.conv2d(conv3_2, filters=256, kernel_size=(3, 3), padding='same',
                           activation=tf.nn.relu, use_bias=True, name='conv3_3')
conv3_4 = tf.layers.conv2d(conv3_3, filters=256, kernel_size=(3, 3), padding='same',
                           activation=tf.nn.relu, use_bias=True, name='conv3_4')
pool3 = tf.layers.max_pooling2d(conv3_3, pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')
```

Clearly, a lot of hyperparameters tend to repeat themselves, since hyper-parameters are usually chosen at the granularity 
of the whole architecture (e.g. whether padding or not in convolutional layers, the choice of the non-linearity, the kernel size etc.).

So it may be worth defining two partials to encapsulate all these shared hyper-parameters:

```python
conv2d_vgg = functools.partial(tf.layers.conv2d, kernel_size=(3, 3), padding='same',
                               activation=tf.nn.relu, use_bias=True)
pool2d_vgg = functools.partial(tf.layers.max_pooling2d, pool_size=(2, 2), strides=(2, 2), padding='same')
```

Now the definition of the first three blocks of VGG19 boils down to the following:

```python
conv1_1 = conv2d_vgg(image, filters=64, name='conv1_1')
conv1_2 = conv2d_vgg(conv1_1, filters=64, name='conv1_2')
pool1 = pool2d_vgg(conv1_2, name='pool1')

conv2_1 = conv2d_vgg(pool1, filters=128, name='conv2_1')
conv2_2 = conv2d_vgg(conv2_1, filters=128, name='conv2_2')
pool2 = pool2d_vgg(conv2_2, name='pool2')

conv3_1 = conv2d_vgg(pool2, filters=256, name='conv3_1')
conv3_2 = conv2d_vgg(conv3_1, filters=256, name='conv3_2')
conv3_3 = conv2d_vgg(conv3_2, filters=256, name='conv3_3')
conv3_4 = conv2d_vgg(conv3_3, filters=256, name='conv3_4')
pool3 = pool2d_vgg(conv3_4, name='pool3')
```

Cool, right? This is just an example. Clearly there are other ways to get the same result, but I find this one particularly elegant. That’s all.
