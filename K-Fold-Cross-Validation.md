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
