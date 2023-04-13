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


