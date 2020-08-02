# Week 1
`#End-to-End` `#Classification`

##

### Chapters
- Chapter 2: End-to_End Machine Learning Project
- Chapter 3: Classification 

##

### Practice Codes
- [CH02](./codes/CH02_end_to_end.ipynb)
- [CH03](./codes/CH03_classification.ipynb)

##

### Takeaways
- <code>**Series.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)**</code>

  **normalize** = If True then the object returned will contain the relative frequencies of the unique values\
  **sort** = Sort by frequencies\
  **bins** = Rather than count values, group them into half-open bins, a convenience for <code>pd.cut</code>, only works with numeric data.
  
  ```
  >>> index = pd.Index([3, 1, 2, 3, 4, np.nan])
  >>> index.value_counts()
  3.0    2
  4.0    1
  2.0    1
  1.0    1
  dtype: int64
  >>> s = pd.Series([3, 1, 2, 3, 4, np.nan])
  >>> s.value_counts(normalize=True)
  3.0    0.4
  4.0    0.2
  2.0    0.2
  1.0    0.2
  dtype: float64
  
  ```
- <code>**DataFrame[DataFrame.isnull().any(axis=1)]**</code>
  
  Returns all rows in dataframe with at least one null value.
  
- <code>**DataFrame.iloc**</code>

  Purely integer-location based indexing for selection by position
  
  Allowed inputs are:
  - An integer, e.g. 5
  - A list or array of integers, e.g. [4, 3, 0]
  - A slice object with ints, e.g. 1:7
  - A boolean array
  
  ```
  >>> mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
  ... {'a': 100, 'b': 200, 'c': 300, 'd': 400},
  ... {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
  >>> df = pd.DataFrame(mydict)
  >>> df
        a     b     c     d
  0     1     2     3     4
  1   100   200   300   400
  2  1000  2000  3000  4000
  ```
  
  
- <code>**DataFrame.loc**</code>

  Access a group of rows and columns by labels or a boolean array

  Allowed inputs are:
  - A single label, e.g. 5 or 'a'
  - A list or array of labels, e.g. ['a', 'b', 'c']
  - A slice object with labels, e.g. 'a':'f'

  ```
  >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
  ... index=['cobra', 'viper', 'sidewinder'],
  ... columns=['max_speed', 'shield'])
  >>> df
              max_speed  shield
  cobra               1       2
  viper               4       5
  sidewinder          7       8
  ```

- <code>**sklearn.linear_model.SGDClassifier(loss='hinge', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)**</code>

  - Linear classifiers (SVM, logistic regression, etc.) with SGD training
  - For best results using the default learning rate schedule, the data should have zero mean and unit variance
  - This implementation works with data represented as dense or sparse arrays of floating point values for the features
  - The model it fits can be controlled with the loss parameter; by default, it fits a linear support vector machine (SVM)
  
  ```
  >>> import numpy as np
  >>> from sklearn.linear_model import SGDClassifier
  >>> from sklearn.preprocessing import StandardScaler
  >>> from sklearn.pipeline import make_pipeline
  >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
  >>> Y = np.array([1, 1, 2, 2])
  >>> # Always scale the input. The most convenient way is to use a pipeline.
  >>> clf = make_pipeline(StandardScaler(),
  ...                     SGDClassifier(max_iter=1000, tol=1e-3))
  >>> clf.fit(X, Y)
  Pipeline(steps=[('standardscaler', StandardScaler()),
                  ('sgdclassifier', SGDClassifier())])
  >>> print(clf.predict([[-0.8, -1]]))
  [1]
  ```
  
- <code>**sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)**</code>

  - Classifier implementing the k-nearest neighbors vote
  - Can be used for multilabel classification
  
  ```
  >>> from sklearn.neighbors import KNeighborsClassifier
  >>> y_train_large = (y_train >= 7)
  >>> y_train_odd = (y_train % 2 == 1)
  >>> y_multilabel = np.c_[y_train_large, y_train_odd]

  >>> knn_clf = KNeighborsClassifier()
  >>> knn_clf.fit(X_train, y_multilabel)
  >>> knn_clf.predict([some_digit])
  array([[False,  True]])
  ```
