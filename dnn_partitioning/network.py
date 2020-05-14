import tensorflow as tf 
from tensorflow.keras.layers import add, Concatenate, Add, Dropout, Softmax, Flatten, Input, Conv2D, MaxPool2D, Activation,Dense,BatchNormalization, AveragePooling2D
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model

#%%

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def ResNet20(input_shape, depth, num_classes=10):

    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)

        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation=None, #'softmax',
                    kernel_initializer='he_normal')(y)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def ResNet20_B(input_shape, depth, num_classes=10):

    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        
        
        if stack==0:
            # ----- FIRST BRANCH ------
            exit1 = x
            exit1 = Conv2D(num_filters, 3, padding="valid")(exit1)
            exit1 = Flatten()(exit1)
            exit1 = Dense(num_classes)(exit1)
        elif stack==1:
            # ----- SECOND BRANCH ------
            exit2 = x 
            exit2 = Flatten()(exit2)
            exit2 = Dense(num_classes)(exit2)

        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    y = Dense(num_classes,
                    activation=None, #'softmax',
                    kernel_initializer='he_normal')(y)
    outputs = [exit1, exit2, y]
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model



# model = ResNet20_B(input_shape=(32,32,3),
#                  depth=20,
#                  num_classes=10)

# tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)




# class ResnetLayer(tf.keras.layers.layer):
#     def __init__(self, 
#                  num_filters=16,
#                  kernel_size=3,
#                  strides=1,
#                  activation='relu',
#                  batch_normalization=True,
#                  conv_first=True):
#         super(ResnetLayer, self).__init()
#         self.conv = Conv2D(num_filters,
#                   kernel_size=kernel_size,
#                   strides=strides,
#                   padding='same',
#                   kernel_initializer='he_normal',
#                   kernel_regularizer=l2(1e-4))
        
#     def call(self, inputs):
#         x = inputs
#         if self.conv_first:
#             x = self.conv(x)
#             if self.batch_normalization:
#                 x = self.BatchNormalization()(x)
#             if self.activation is not None:
#                 x = Activation(self.activation)(x)
#         else:
#             if self.batch_normalization:
#                 x = BatchNormalization()(x)
#             if self.activation is not None:
#                 x = Activation(self.activation)(x)
#             x = self.conv(x)
#         return x


# x = ResnetLayer()



def branchyNet_6 (input_shape=(32,32,3), num_classes=10):
    
    conv1 = Conv2D(96,3, activation='relu', padding='same')
    conv2 = Conv2D(96,3, activation='relu', padding='same')
    conv3 = Conv2D(96,3, strides=2, activation='relu', padding='same')

    conv4 = Conv2D(192,3, activation='relu', padding='same')
    conv5 = Conv2D(192,3, activation='relu', padding='same')
    conv6 = Conv2D(192,3, strides=2, activation='relu', padding='same')

    conv7 = Conv2D(192,3, activation='relu', padding='same')
    conv8 = Conv2D(192,3, activation='relu', padding='same')
    conv9 = Conv2D(192,3, strides=2, activation='relu', padding='same')

    conv10 = Conv2D(10,3, activation='relu', padding='same')

    conv11 = Conv2D(96,3, activation='relu', padding='valid')
    conv12 = Conv2D(96,3, activation='relu', padding='valid')

    conv21 = Conv2D(96,3, activation='relu', padding='valid')

    averpool = AveragePooling2D(pool_size=4, strides=4, padding="valid")
    flatten1 = Flatten() 
    flatten2 = Flatten() 
    flatten3 = Flatten() 
    drop1 = Dropout(0.1)
    drop2 = Dropout(0.1)
    drop3 = Dropout(0.1)
    drop4 = Dropout(0.1) 
    d1 = Dense(10)
    d2 = Dense(10)

    inputs = Input(shape=input_shape)
    x = drop1(inputs)
    x1 = conv1(x)
    x = conv2(x1)
    x = conv3(x)
    x2 = drop2(x)
    x = conv4(x2)
    x = conv5(x)
