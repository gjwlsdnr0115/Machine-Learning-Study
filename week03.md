# Week 3
`#DecisionTree` `#EnsembleLearning` `#RandomForest`

##

### Chapters
- Chapter 6: Decision Trees
- Chapter 7: Ensemble Learning and Random Forests

##

### Practice Codes
- [CH06](./codes/CH06_decision_trees.ipynb)
- [CH07](./codes/CH07_ensemble_learning_and_random_forests.ipynb)

##

### Takeaways
- <code>**sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0)**</code>
  
  **criterion** = The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.\
  **max_depth** = The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.\
  **min_samples_split** = The minimum number of samples required to split an internal node.\
  **min_samples_leaf** = The minimum number of samples required to be at a leaf node.\
  **max_features** = The number of features to consider when looking for the best split.\
  
  
  ```
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.model_selection import cross_val_score
  >>> from sklearn.tree import DecisionTreeClassifier
  >>> clf = DecisionTreeClassifier(random_state=0)
  >>> iris = load_iris()
  >>> cross_val_score(clf, iris.data, iris.target, cv=10)
  ...                             # doctest: +SKIP
  ...
  array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
          0.93...,  0.93...,  1.     ,  0.93...,  1.      ])  
  ```
  
- <code>**sklearn.tree.DecisionTreeRegressor(*, criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort='deprecated', ccp_alpha=0.0)**</code>
  
  Decision tree for regression
  
  ```
  >>> from sklearn.datasets import load_diabetes
  >>> from sklearn.model_selection import cross_val_score
  >>> from sklearn.tree import DecisionTreeRegressor
  >>> X, y = load_diabetes(return_X_y=True)
  >>> regressor = DecisionTreeRegressor(random_state=0)
  >>> cross_val_score(regressor, X, y, cv=10)
  ...                    # doctest: +SKIP
  ...
  array([-0.39..., -0.46...,  0.02...,  0.06..., -0.50...,
         0.16...,  0.11..., -0.73..., -0.30..., -0.00...])
  ```
  
  
- <code>**sklearn.ensemble.RandomForestClassifier()**</code>

  A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

  **n_estimators** = The number of trees in the forest.\
  **bootstrap** = Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.\
  **max_samples** = If bootstrap is True, the number of samples to draw from X to train each base estimator.
  
  ```
  >>> from sklearn.ensemble import RandomForestClassifier
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=1000, n_features=4,
  ...                            n_informative=2, n_redundant=0,
  ...                            random_state=0, shuffle=False)
  >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
  >>> clf.fit(X, y)
  RandomForestClassifier(...)
  >>> print(clf.predict([[0, 0, 0, 0]]))
  [1]
  ```
  
  
- <code>**sklearn.ensemble.BaggingClassifier()**</code>

  An ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.\
  Can be used as a way to reduce the variance of a black-box estimator by introducing randomization into its construction procedure and then making an ensemble out of it.

  ```
  >>> from sklearn.svm import SVC
  >>> from sklearn.ensemble import BaggingClassifier
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=100, n_features=4,
  ...                            n_informative=2, n_redundant=0,
  ...                            random_state=0, shuffle=False)
  >>> clf = BaggingClassifier(base_estimator=SVC(),
  ...                         n_estimators=10, random_state=0).fit(X, y)
  >>> clf.predict([[0, 0, 0, 0]])
  array([1])
  ```

- <code>**xgboost.XGBRegressor()**</code>

  Implementation of the scikit-learn API for XGBoost regression.
  
  ```
  >>> xgb_reg = xgboost.XGBRegressor(random_state=42)
  >>> xgb_reg.fit(X_train, y_train)
  >>> y_pred = xgb_reg.predict(X_val)
  >>> val_error = mean_squared_error(y_val, y_pred) # Not shown
  >>> print("Validation MSE:", val_error)
  Validation MSE: 0.0028512559726563943
  ```
