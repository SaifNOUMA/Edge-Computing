#%% 
import os
from network1 import ResNet20_B
import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 50
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

model = ResNet20_B(input_shape=(32,32,3),depth=20)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

checkpoint_directory = "checkpoints/training_checkpoints_resnet_B_v1/"
checkpoint_prefix = os.path.join(checkpoint_directory)

checkpoint = tf.train.Checkpoint(optimzer= optimizer, model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))


test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy1 = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy_1')
test_accuracy2 = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy_2')
test_accuracy3 = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy_3')

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  # pred1 , pred2 , pred3 = predictions[:,:10] , predictions[:,10:20] , predictions[:,20:]
  pred1 , pred2 , pred3 = predictions[0] , predictions[1] , predictions[2]
  loss1 , loss2 , loss3 = loss_object(labels, pred1) , loss_object(labels, pred2) , loss_object(labels, pred3)
  t_loss = loss1 + loss2 + loss3

  test_loss(t_loss)
  test_accuracy1(labels, pred1)
  test_accuracy2(labels, pred2)
  test_accuracy3(labels, pred3)

for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

template = 'Test Accuracy 1: {}, Test Accuracy 2: {}, Test Accuracy 3: {}'
print(template.format(test_loss.result(),
                    test_accuracy1.result() * 100,
                    test_accuracy2.result() * 100,
                    test_accuracy3.result() * 100))


