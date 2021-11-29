import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow._api.v2 import data

dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=(None, ))

for batch in dataset.take(3):
    print(batch.numpy())
    print()


# Processing multiple epochs

titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)

def plot_batch_sizes(ds):
    batch_sizes = [batch.shape[0] for batch in ds]
    plt.bar(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('Batch number')
    plt.ylabel('Batch size')
    plt.show()

titanic_batches = titanic_lines.repeat(3).batch(128)
plot_batch_sizes(titanic_batches)

titanic_batches = titanic_lines.batch(128).repeat(3)
plot_batch_sizes(titanic_batches)


# Custom computation in the end of each epoch

epochs = 3
dataset = titanic_lines.batch(128)

for epoch in range(epochs):
    for batch in dataset:
        print(batch.shape)
    print("End of epoch: ", epoch)


# Randomly shuffling input data

lines = tf.data.TextLineDataset(titanic_file)
counter = tf.data.experimental.Counter()

dataset = tf.data.Dataset.zip((counter, lines))
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(20)
dataset

dataset = tf.data.Dataset.zip((counter, lines))
shuffled = dataset.shuffle(buffer_size=100).batch(10). repeat(2)
for n, line_batch in shuffled.skip(60).take(5):
        print(n.numpy())

