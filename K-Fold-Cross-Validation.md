### Using k-fold in the simpleDL network

#### study material

[StatQuest](https://www.youtube.com/watch?v=fSytzGwwBVw&ab_channel=StatQuestwithJoshStarmer)

[codebasics](https://www.youtube.com/watch?v=gJo0uNL-5Qw&ab_channel=codebasics)

[colab](https://www.kaggle.com/code/robikscube/cross-validation-visualized-youtube-tutorial/notebook)

* use k fold to have the training and testing indexes
* for each group of indexes you train using training_data[k_training_indices]
* save the results
* take an average at the end

#### k = 10, ephocs = 50

```python
n_split = 10
skf = StratifiedKFold(n_splits=n_split, random_state=42, shuffle=True)
accuracys = []

k = 0
for (train_index, test_index) in skf.split(trainingImages, labels):
    
  model = simple_dl()

  callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  print(f'\n\n\t FOLD: {k+1}/{n_split}', end="\n\n")

  model.fit(trainingImages[train_index], labels[train_index], epochs=50, \
            validation_data=(trainingImages[test_index], labels[test_index]), callbacks=[callback])

  loss, accuracy = model.evaluate(trainingImages[test_index], labels[test_index])
  accuracys.append(accuracy)
    
  k+=1
print("\n\nAccuracys : ", accuracys, end="\n\n")
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(accuracys), np.std(accuracys)))

```

Accuracys :  [0.699, 0.850, 0.699, 0.850, 0.75, 0.75, 0.75, 0.800, 0.850, 0.850]

Cross-Validation accuracy: 0.785 +/- 0.059

### Data Augmentation

[DeepLizard](https://www.youtube.com/watch?v=WSvpLUietIM&ab_channel=deeplizard)

[DigitalSreeni](https://www.youtube.com/watch?v=ccdssX4rIh8&ab_channel=DigitalSreeni)

### k = 10, flip, factor rotation 0.05, fill_mode='reflect', zoom = 0.05
```python
dataAugmentation = tf.keras.models.Sequential([
    
  tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", input_shape=(size,size,1)),

  tf.keras.layers.RandomRotation(factor=0.05,fill_mode='reflect'),
  
  tf.keras.layers.RandomZoom(0.05)]
)
```
Accuracys :  [0.800000011920929, 0.75, 0.6499999761581421, 0.800000011920929, 0.800000011920929, 0.699999988079071, 0.6499999761581421, 0.75, 0.44999998807907104, 0.6000000238418579]

Cross-Validation accuracy: 0.695 +/- 0.106

### k = 10, flip, factor rotation 0.05, fill_mode='nearest', zoom = 0.05

Accuracys :  [0.550000011920929, 0.699999988079071, 0.6000000238418579, 0.75, 0.8500000238418579, 0.75, 0.800000011920929, 0.75, 0.6499999761581421, 0.75]

Cross-Validation accuracy: 0.715 +/- 0.087

### k = 5, factor rotation 0.05, fill_mode='nearest', zoom = 0.05

Accuracys :  [0.7749999761581421, 0.699999988079071, 0.7749999761581421, 0.824999988079071, 0.8500000238418579]

Cross-Validation accuracy: 0.785 +/- 0.051

### k = 10, factor rotation 0.05, fill_mode='nearest', zoom = 0.05

Accuracys :  [0.699999988079071, 0.8500000238418579, 0.699999988079071, 0.800000011920929, 0.8500000238418579, 0.699999988079071, 0.6499999761581421, 0.949999988079071, 0.800000011920929, 0.800000011920929]

Cross-Validation accuracy: 0.780 +/- 0.087

### batch_size = 64, k = 5, factor rotation 0.05, fill_mode='nearest', zoom = 0.05

Accuracys :  [0.824999988079071, 0.7749999761581421, 0.824999988079071, 0.8500000238418579, 0.8500000238418579]

Cross-Validation accuracy: 0.825 +/- 0.027

### batch_size = 128, k = 5, factor rotation 0.05, fill_mode='nearest', zoom = 0.05

Accuracys :  [0.75, 0.20000000298023224, 0.8500000238418579, 0.8500000238418579, 0.8500000238418579]
Cross-Validation accuracy: 0.700 +/- 0.253

### batch_size = 64, epochs=80 k = 5, flip, factor rotation 0.05, fill_mode='nearest', zoom = 0.05

Accuracys :  [0.75, 0.20000000298023224, 0.8500000238418579, 0.8500000238418579, 0.8500000238418579]
Accuracys :  [0.6499999761581421, 0.675000011920929, 0.675000011920929, 0.7749999761581421, 0.7749999761581421]

Cross-Validation accuracy: 0.710 +/- 0.054


### batch_size = 64, epochs=80 k = 5, factor rotation 0.1, fill_mode='nearest', zoom = 0.05

Accuracys :  [0.625, 0.7749999761581421, 0.800000011920929, 0.800000011920929, 0.7749999761581421]

Cross-Validation accuracy: 0.755 +/- 0.066

### batch_size = 64, epochs=80 k = 5, factor rotation 0.05, fill_mode='nearest', zoom = 0.1

Accuracys :  [0.8500000238418579, 0.7250000238418579, 0.8500000238418579, 0.824999988079071, 0.875]

Cross-Validation accuracy: 0.825 +/- 0.052
