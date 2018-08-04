# Why `Functools.partial` may declutter your code

### Motivation
If you're a code-writing kind of guy, from time to time you may have found yourself in the situation in which you have to call the same function multiple times, with only a small or no variation in the parameters. So in multiple places you have this *looong* signature with a *loots* of parameters, although maybe 90% of them are just always the same.

And here is where [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial) may come to the rescue.

### Explanation

In a nutshell, [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial) *is used to “freeze” some portion of a function’s arguments and/or keywords resulting in a new object 
with a simplified signature*.

Borrowing from the docs:
> `functools.partial(func, *args, **keywords)` Return a new partial object which when called will behave like *func* called with the positional arguments *args* and keyword arguments *keywords*. If more arguments are supplied to the call, they are appended to args. If additional keyword arguments are supplied, they extend and override *keywords*.

This is roughly equivalent to:
```python
def partial(func, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return func(*args, *fargs, **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc
```

### Example 

So, let's see a real-world example to see how this may be useful.

Since I work a lot with deep networks, the first example that came to my mind is related to the topic. But relax, to our purpose no domain knowledge is necessary. The example explains by itself.

Let's say we want to define a convolutional neural network in TensorFlow. By the way, the following are the first three block of layers of VGG19 network, if you know what I'm talking about. But if you're not really into the field and you don't know what a convolution is, that's totally fine too. For our aim, the only important thing is that we have to repeatedly use the `tf.layers.conv2d` function with a specific set of arguments.

Here is how our code looks like at this point:
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
As it should be evident, a lot of parameters tend to repeat themselves. From the technical point of view, in this case this is due to the fact that hyper-parameters are usually chosen at the granularity of the whole network, so all layers have the same. But whathever the reason is, the truth is that our code looks really cluttered and redundant.

This is when it may be worth defining two partials to encapsulate all these shared hyper-parameters:

```python
conv2d_vgg = functools.partial(tf.layers.conv2d, kernel_size=(3, 3), padding='same',
                               activation=tf.nn.relu, use_bias=True)
pool2d_vgg = functools.partial(tf.layers.max_pooling2d, pool_size=(2, 2), strides=(2, 2), padding='same')
```

Now, the definition of the very same network boils down to the following:

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

*Cool, right?* And I bet there are many other examples that come to your mind!

Clearly there are other ways to get a similar result. For example I could have defined a function:
```python
def conv2d_vgg(input, name):
    return tf.layers.conv2d(input, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
    use_bias=True, name=name)
```
but this would have taken more lines, and then maybe it would end up in a different file or module, forcing the reader to look for it somewhere else to understand what's going on. Instead, using `functools.partial` as above allows to declutter the code while keeping everything in the same place. So overall I find it is a particularly elegant solution. What's your take on this?
