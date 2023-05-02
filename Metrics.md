### Top-K accuracy

Top-k accuracy is a metric used in machine learning to evaluate how well a classification model performs on a multi-class classification problem. Instead of just measuring whether a model predicted the correct class, top-k accuracy measures whether the model predicted the correct class within the top k most likely classes.

I used sparse_top_k_categorical_accuracy cause the labels are integer-encoded

### F-Score
See before:

* precision
* recall
* completeness
* 


F-score is a metric used to evaluate the performance of a binary or multi-class classification model. It is a combination of **precision** and **recall**, which are metrics that measure the **accuracy** and **completeness** of the model's predictions, respectively.

There are two common types of F-score: micro and macro.

Micro F-score is calculated by aggregating the total true positives, false positives, and false negatives across all classes, and then computing the precision, recall, and F-score for the aggregated data. Micro F-score gives equal weight to each sample and each class, and is typically used when the classes are imbalanced.

Macro F-score is calculated by computing the precision, recall, and F-score for each class, and then averaging the scores across all classes. Macro F-score gives equal weight to each class, regardless of the number of samples in each class, and is typically used when the classes are balanced.

In general, micro F-score is more suitable when the dataset is imbalanced, while macro F-score is more suitable when the dataset is balanced.
