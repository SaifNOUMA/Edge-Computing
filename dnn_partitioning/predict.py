#%% 
from utils import *
from resnet import *

depth , version = 20 , 1
subtract_pixel_mean, num_classes = True, 10
# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

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

model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss={"Exit_1" : 'categorical_crossentropy', "Exit_2" : 'categorical_crossentropy', "Exit_3" : 'categorical_crossentropy'},
              loss_weights={"Exit_1" : 1, "Exit_2" : 1, "Exit_3" : 1},
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])

keras.utils.plot_model(model, show_shapes=True, dpi=64)

print(get_weights()[-1])
model.load_weights(get_weights()[-1])

#%%
# Score trained model.
# scores = model.evaluate(x_test,  {"Exit_1" : y_test, "Exit_2" : y_test, "Exit_3" : y_test}, verbose=1)
# print(scores)
# print('Test loss:', scores[0])
# print('--------------------   Test accuracy:  --------------------\nExit 1: %f\nExit 2: %f\nExit 3: %f\n' % (scores[-3], scores[-2], scores[-1]) )

input_, output_ = model.input , model.output

m_exit1 , m_exit2, m_exit3 = Model(input_, output_[0]) , Model(input_, output_[1]) , Model(input_, output_[2]) 
m_exit1.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])
m_exit2.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
m_exit3.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
              
# keras.utils.plot_model(m_exit1, show_shapes=True, dpi=64)

#%% 
m_exit2_t = m_exit2_((32,32,16), depth=depth)
for l1 , l2 in zip(m_exit2.layers[-24:],m_exit2_t.layers[1:]):
    # print("------------ Original ------------")
    # print(l1.get_config())
    # print("------------ Fake ------------")
    # print(l2.get_config())
    # time.sleep(2)
    l2.set_weights(l1.get_weights())
#%%
m_exit3_t = m_exit3_((16,16,32), depth=depth)
for l1 , l2 in zip(m_exit3.layers[-25:],m_exit3_t.layers[1:]):
    l2.set_weights(l1.get_weights())
    # print("------------ Original ------------")
    # print(l1.get_config())
    # print("------------ Fake ------------")
    # print(l2.get_config())
    # time.sleep(2)

#%%
m_exit1 = Model(m_exit1.input, [m_exit1.layers[-4].output,m_exit1.output])
m_exit2 = m_exit2_t
m_exit3 = m_exit3_t

m_exit1.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])
m_exit2.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
m_exit3.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
              
keras.utils.plot_model(m_exit1, show_shapes=True, dpi=64, to_file='results/plots/exit1.png')
keras.utils.plot_model(m_exit2, show_shapes=True, dpi=64, to_file='results/plots/exit2.png')
keras.utils.plot_model(m_exit3, show_shapes=True, dpi=64, to_file='results/plots/exit3.png')


#%% 

m_exit1.save("weights/exit1.h5")
m_exit2.save("weights/exit2.h5")
m_exit3.save("weights/exit3.h5")

# Prediction for a single exit
# scores = m_exit1.evaluate(x_test,  y_test, verbose=1)
# print(scores)
# print('Test loss:', scores[0], '\nTest Accuracy:', scores[1])


# %%
