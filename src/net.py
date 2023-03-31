from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *



def cnn_ascad_fix_ID(length, lr=0.00001, classes=256):
    img_input = Input(shape=(length, 1))
    x = Conv1D(128, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = Adam(lr=5e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def cnn_ascad_variable_ID(length, lr=0.00001, classes=256):
    img_input = Input(shape=(length, 1))
    x = Conv1D(128, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = Adam(lr=5e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model



def cnn_chipw(length, lr=0.0001, classes=256):
    # From VGG16 design
    input_shape = (length, 1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(8, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_chipw')
    model.summary()
    optimizer = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def cnn_aes_HD_fix_ID(length, lr=0.00001, classes=256):
    img_input = Input(shape=(length, 1))
    x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='aes_hd_model')
    optimizer = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

