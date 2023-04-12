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
using softmax activation
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
