# training-notes
 image recognition neural network training notes for my research

### 2023-04-12
```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(size, (3,3), activation='relu', input_shape=(size,size,1)))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))
```
loss: 2.0531 - accuracy: 0.6970

### 2023-04-12
using softmax activation and size of image = 250
```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(size, (3,3), activation='relu', input_shape=(size,size,1)))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
```
loss: 1.5270 - accuracy: 0.7424

### 2023-04-12
using one more convLayer and ephoc = 20
```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(size, (3,3), activation='relu', input_shape=(size,size,1)))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
```
loss: 1.4661 - accuracy: 0.7424

### 2023-04-12
changing the filters size in conv layers

```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(size, (3,3), activation='relu', input_shape=(size,size,1)))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
```
loss: 0.9315 - accuracy: 0.7727

### 2023-04-13
#### Notes on the [tuning playbook](https://github.com/google-research/tuning_playbook) that may be useful for the project.

#### Batch-size:
Often, the ideal batch size will be the largest batch size supported by the available hardware.

#### Checkpoint model or weights:
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

#### See later 
https://www.tensorflow.org/guide/profiler?hl=pt-br

https://github.com/google-research/interpretability-theory

https://www.kaggle.com/code/shivamb/cnn-architectures-vgg-resnet-inception-tl

### Training CNN Architectures : AlexNet, VGG, ResNet, Inception
#### AlexNet

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(size,size,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
model.summary()

model.fit(training_images, training_labels, epochs=50, validation_data=(testing_images, testing_labels), validation_freq=1)
```
loss: 1.7844 - accuracy: 0.4091

#### VGG19

```python
model = tf.keras.models.Sequential()

# Block 1
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(size, size, 1)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

# Block 2
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

# Block 3
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

# Block 4
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

# Block 5
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

# Classification block
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=25, validation_data=(testing_images, testing_labels))
```
loss: 2.2375 - accuracy: 0.1515

#### InceptionNet

```python
model = tf.keras.models.Sequential()

# First Inception block
model.add(tf.keras.layers.Conv2D(64, (1,1), padding='same', activation='relu', input_shape=(size, size, 1)))
model.add(tf.keras.layers.Conv2D(96, (1,1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(16, (1,1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'))
model.add(tf.keras.layers.Conv2D(32, (1,1), padding='same', activation='relu'))

# Second Inception block
model.add(tf.keras.layers.Conv2D(128, (1,1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (1,1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(192, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(32, (1,1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(96, (5,5), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'))
model.add(tf.keras.layers.Conv2D(64, (1,1), padding='same', activation='relu'))

# Output layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))
```
loss: 2.2358 - accuracy: 0.1515



