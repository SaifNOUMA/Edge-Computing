#%% 
from utils import *
from keras.models import load_model

subtract_pixel_mean, num_classes = True, 10

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


m_exit1 , m_exit2 , m_exit3 = load_model("weights/exit1.h5") , load_model("weights/exit2.h5") , load_model("weights/exit3.h5")

#%%  Prediction for Exit 1 || Compute the intermediate vector
runtime = []
import time
t0 = time.time()
y_pred1 = m_exit1.predict(x_test[0:1])
y_pred = np.argmax(y_pred1[1], axis=-1)
t1 = time.time()
runtime.append(t1-t0)
# accuracy1 = 100*np.sum(y_pred==np.argmax(y_test,axis=-1))/len(y_test)
# print(accuracy1)

#%%
import tensorflow as tf
class MyCustomCallback(tf.keras.callbacks.Callback):

   def on_predict_begin(self, batch, logs=None):
        set_batch_time(time.time_ns())

   def on_predict_end(self, batch, logs=None):
        curr_time = time.time_ns()

        diff_time = curr_time - get_batch_time()
        print('Predict {} duration: {}'.format(self.model.layers[1].name, diff_time))

pred = m_exit1.predict(x_test, batch_size=256, verbose=1, callbacks=[MyCustomCallback()])


#%%  Prediction for Exit 2 || Compute the intermediate vector
t0 = time.time()
y_pred2 = m_exit2.predict(x_test[0:1])
y_pred = np.argmax(y_pred2, axis=-1)
t1 = time.time()
runtime.append(t1-t0)
# accuracy2 = 100*np.sum(y_pred==np.argmax(y_test,axis=-1))/len(y_test)
# print(accuracy2)

#%% Prediction for Exit 3 || Compute the intermediate vector
t0 = time.time()
y_pred3 = m_exit3.predict(x_test[0:1])
y_pred = np.argmax(y_pred3, axis=-1)
t1 = time.time()
runtime.append(t1-t0)
# accuracy3 = 100*np.sum(y_pred==np.argmax(y_test,axis=-1))/len(y_test)
# print(accuracy3)

#%% 

runtime = [100*i for  i in runtime]

#%%
# m_exit1.compile(loss='categorical_crossentropy',
#               optimizer=Adam(),
#               metrics=['accuracy'])
# m_exit2.compile(loss='categorical_crossentropy',
#               optimizer=Adam(),
#               metrics=['accuracy'])
# m_exit3.compile(loss='categorical_crossentropy',
#               optimizer=Adam(),
#               metrics=['accuracy'])

# for ex, model in enumerate([m_exit1,m_exit2,m_exit3]):
#     y_pred = model.predict(x_train)
#     log_y_pred = np.ma.log(y_pred).filled(0)

#     entropies_train = -np.sum(np.multiply(y_pred,log_y_pred), axis=-1)
#     # plt.title("Entropy of train samples for Exit %d" % (ex))
#     # plt.savefig("results/plots/entropy_train_exit_%d.png" % ex)

#     y_pred = model.predict(x_test)
#     log_y_pred = np.ma.log(y_pred).filled(0)

#     entropies_test = -np.sum(np.multiply(y_pred,log_y_pred), axis=-1)

#     # bins = [x + 0.5 for x in range(0, 6)]
#     plt.hist([entropies_train, entropies_test], bins=5, color = ['green', 'blue'], label=['train','test'])
#     plt.xlabel("entropy")
#     plt.ylabel("count")
#     plt.title("Entropy for Exit %d" % (ex+1))
#     plt.legend()
#     plt.savefig("results/plots/entropy_exit_%d.png" % (ex+1))
#     plt.close()



#%%
for ex, model in enumerate([m_exit1,m_exit2,m_exit3]):

    y_pred = model.predict(x_test)
    log_y_pred = np.ma.log(y_pred).filled(0)

    y_pred_label = np.argmax(y_pred, axis=-1)
    y_test_label = np.argmax(y_test, axis=-1)

    y_true = y_pred[y_pred_label==y_test_label]
    log_y_true = log_y_pred[y_pred_label==y_test_label]

    y_false = y_pred[y_pred_label!=y_test_label]
    log_y_false = log_y_pred[y_pred_label!=y_test_label]

    entropies_true = -np.sum(np.multiply(y_true,log_y_true), axis=-1)
    entropies_false = -np.sum(np.multiply(y_false,log_y_false), axis=-1)

    plt.hist(x = [entropies_true, entropies_false], 
            bins = [0.05*i for i in range(1,21)],
            color = ['green', 'blue'], 
            label=['True','False'])
    plt.xlabel("entropy")
    plt.ylabel("count")
    plt.title("Entropy for Exit %d" % (ex+1))
    plt.legend()
    plt.savefig("results/plots/entropy_exit_test_%d.png" % (ex+1))
    plt.close()

#%%

# Prediction for a single exit
# scores = m_exit1.evaluate(x_test,  y_test, verbose=1)
# print(scores)
# print('Test loss:', scores[0], '\nTest Accuracy:', scores[1])
# scores = m_exit2.evaluate(x_test,  y_test, verbose=1)
# print(scores)
# print('Test loss:', scores[0], '\nTest Accuracy:', scores[1])
# scores = m_exit3.evaluate(x_test,  y_test, verbose=1)
# print(scores)
# print('Test loss:', scores[0], '\nTest Accuracy:', scores[1])



# %%
# from keras.layers import Input
# m_exit2_input = Input((32, 32, 16))
# m_exit2_output = m_exit2_input
# for layer in m_exit2.layers[25:]:
#     m_exit2_output = layer(m_exit2_output)
# m_exit21 = Model(inputs=m_exit2_input, outputs=m_exit2_output)