# Hi
I've been busy with final exams these days so basically these codes constructs classic models with keras and benchmark them.  
Will update fasternet code soon.
## Update #1
I've uploaded pconv, a partial convolution layer coded in keras. You can simply use this Convolution layer like this:

```
import tensorflow as tf
from tensorflow import keras
import pconv

# loading mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# data pretrain
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = keras.Sequential([
    # PConv
    pconv.PConv2D(dim=1, n_div=1),
    pconv.MLPBlock(dim=1, hidden_dim=4),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    # 10 categories
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

```

# About Benchmark
- Prepare imagenet dataset
- Manually replace dataset filepath in keras_(model name).py
- run statistic.py
