### Top-K accuracy

Top-k accuracy is a metric used in machine learning to evaluate how well a classification model performs on a multi-class classification problem. Instead of just measuring whether a model predicted the correct class, top-k accuracy measures whether the model predicted the correct class within the top k most likely classes.

I used sparse_top_k_categorical_accuracy cause the labels are integer-encoded

### Precision and Recall

https://deepai.org/machine-learning-glossary-and-terms/precision-and-recall

* Precision = truePositive / (truePositive + falsePositive)

* Recall = truePositive / (truePositive + falseNegative)

![image](https://user-images.githubusercontent.com/86575893/235692142-d6a2665f-a5bb-46e9-8345-35601ad7760f.png)

### Sensitivity and Specificity

* Sensitivity = recall.

Sensitivity measures the proportion of actual positives that are correctly identified as such (tp / (tp + fn)). 

* Specificity = trueNegatives / (trueNegatives + falsePositives)

Specificity measures the proportion of actual negatives that are correctly identified as such (tn / (tn + fp)).

#### Precision, Recall and F-score

"Usually, precision and recall scores are given together and are not quoted individually. This is because it is easy to vary the sensitivity of a model to improve precision at the expense of recall, or vice versa."

### F-Score

harmonic mean of the precision and recall:

![image](https://user-images.githubusercontent.com/86575893/235695249-aa644d4e-4b76-401e-b51d-f7935bf4e417.png)

*F_beta_score* uses diferrent weigths for precision and recall

F-score is a metric used to evaluate the performance of a binary or multi-class classification model. It is a combination of **precision** and **recall**, which are metrics that measure the **accuracy** and **completeness** of the model's predictions, respectively.

There are two common types of F-score: micro and macro.

Micro F-score is calculated by aggregating the total true positives, false positives, and false negatives across all classes, and then computing the precision, recall, and F-score for the aggregated data. Micro F-score gives equal weight to each sample and each class, and is typically used when the classes are imbalanced.

Macro F-score is calculated by computing the precision, recall, and F-score for each class, and then averaging the scores across all classes. Macro F-score gives equal weight to each class, regardless of the number of samples in each class, and is typically used when the classes are balanced.

In general, micro F-score is more suitable when the dataset is imbalanced, while macro F-score is more suitable when the dataset is balanced.
