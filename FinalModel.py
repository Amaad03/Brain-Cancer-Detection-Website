import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, regularizers
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.layers import Input
import os 
import numpy as np
import random


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#We choose to set a random seed for reproductivity
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

#Define directories for training and testing datasets
train_dir = 'dataset/Testing'   # This folder contains the training images with validation split 
test_dir = 'dataset/Training'   # Folder used for evaluating final model 



# Data augmentation 
train_aug = ImageDataGenerator(
    rescale=1./255,           
    rotation_range=30,          
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# For the testing I chose to do simple augmentation since we will evaluate off of this
test_aug = ImageDataGenerator(rescale=1./255)


# For the train_generator:
# it applys the data augmentation
# Resizes images 
# Generate batches of 32 images
# Delegates this as the training subset
# Labels are one-hot encoded (each image belongs to one class and applies a numeric value to represent each)
train_generator = train_aug.flow_from_directory(
    train_dir,
    target_size=(244, 244),
    batch_size=32,
    subset='training',
    class_mode='categorical'
)

validation_generator = train_aug.flow_from_directory(
    train_dir,
    target_size=(244, 244),
    batch_size=32,
    subset='validation',
    class_mode='categorical'
)

test_generator = test_aug.flow_from_directory(
    test_dir,
    target_size=(244, 244),
    batch_size=32,
    class_mode='categorical'
)



#Model Architecture 
# I use 3 convolutional blocks, each consisting of:
# - Two convolution layers followed by BatchNormalization after each convolution
# - MaxPooling2d is applied after each pair of convolutions to downsample the feature maps 


# The Global Average Pooling serves to reduce the output to a 1D vector by averaging over the entire spatial dimensions
# This helps to reduce the number of paramters while perserving the important features


# The Dense layer creates a fully connected layer that connects the previous neurons together and uses ReLU activation
# I use the kernel_regularizer with L2 regularization is applied to help prevent overfitting by penalizing large weights. 

# I apply Dropout to then enable the model to generalize better and stop ovefitting.
# The Final Dense layer, combines all the learned features into a single vector 
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(244,244,3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])




# I used the pre-trained VGG16 Model to be a comparison.
# I excluded the top layer and pre-trained weights in order for it learn off our dataset. 
# I then set the entire base model to be trainable and I freeze the first layers of the base model which help avoid overfitting
# allows the model to have to relarn everything from scratch for a large model like this it is neccessary

'''
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(244, 244, 3))

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

base_model.trainable = True
for layer in base_model.layers[:10]:  # Adjust this number to unfreeze more layers
    layer.trainable = False
'''




# I used the Adam Optimizer for a small learning rate for stable training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


# categorical_crossentropy allows for multi-classs classification
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# early stopping helps prevent overfitting by stopping train when val_loss doesnt improve for 5 epochs
# lr_reduction waits for 2 epochs, and it cuts learning by half and verbose just tells keras to print a message
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('model2.keras', monitor='val_loss', save_best_only=True)

#trains the model, I chose 100 epoch because I want it to keep learning until early_stopping occurs
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100,
    callbacks=[early_stopping, model_checkpoint, lr_reduction],
    verbose=1
)


model.save('vggtest.keras')


test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

plt.figure(figsize=(12, 4))


plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.show()
