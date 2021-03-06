# Week 2
`#Linear` `#Vector`

##

### Chapters
- Chapter 4: Training Linear Models
- Chapter 5: Support Vector Machines

##

### Practice Codes
- [CH04](./codes/CH04_training_models.ipynb)
- [CH05](./codes/CH05_support_vector_machines.ipynb)

##

### Takeaways
- <code>**SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, rho=0.85, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, p=0.1, seed=0, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False)**</code>

  **loss** = The loss function to be used. Defaults to ‘squared_loss’ which refers to the ordinary least squares fit. ‘huber’ is an epsilon insensitive loss function for robust regression.\
  **penalty** = str, ‘l2’ or ‘l1’ or ‘elasticnet’\
  **alpha** = Constant that multiplies the regularization term. Defaults to 0.0001\
  **rho** = The Elastic Net mixing parameter, with 0 < rho <= 1. Defaults to 0.85.\
  **fit_intercept** = Whether the intercept should be estimated or not. If False, the data is assumed to be already centered. Defaults to True.
  
  ```
  >>> import numpy as np
  >>> from sklearn import linear_model
  >>> n_samples, n_features = 10, 5
  >>> np.random.seed(0)
  >>> y = np.random.randn(n_samples)
  >>> X = np.random.randn(n_samples, n_features)
  >>> clf = linear_model.SGDRegressor()
  >>> clf.fit(X, y)
  SGDRegressor(alpha=0.0001, eta0=0.01, fit_intercept=True,
        learning_rate='invscaling', loss='squared_loss', n_iter=5, p=0.1,
        penalty='l2', power_t=0.25, rho=0.85, seed=0, shuffle=False,
        verbose=0, warm_start=False)  
  ```
  
- <code>**Ridge(alpha=1.0, *, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)**</code>
  
  Solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm
  
  ```
  >>> from sklearn.linear_model import Ridge
  >>> ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
  >>> ridge_reg.fit(X, y)
  >>> ridge_reg.predict([[1.5]])
  ```
- <code>**Lasso(alpha=1.0, *, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')**</code>

  Linear Model trained with L1 prior as regularize
  
  ```
  >>> from sklearn.linear_model import Lasso
  >>> lasso_reg = Lasso(alpha=0.1)
  >>> lasso_reg.fit(X, y)
  >>> lasso_reg.predict([[1.5]])
  ```
- <code>**sklearn.preprocessing.StandardScaler(*, copy=True, with_mean=True, with_std=True)**</code>

  Standardize features by removing the mean and scaling to unit variance
  
  The standard score of a sample x is calculated as:

  z = (x - u) / s

  where u is the mean of the training samples or zero if `with_mean=False`, and s is the standard deviation of the training samples or one if `with_std=False`
  
  ```
  >>> from sklearn.preprocessing import StandardScaler
  >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
  >>> scaler = StandardScaler()
  >>> print(scaler.fit(data))
  StandardScaler()
  >>> print(scaler.mean_)
  [0.5 0.5]
  >>> print(scaler.transform(data))
  [[-1. -1.]
  [-1. -1.]
  [ 1.  1.]
  [ 1.  1.]]
  >>> print(scaler.transform([[2, 2]]))
  [[3. 3.]]
  ```

- <code>**LinearSVC(penalty='l2', loss='squared_hinge', *, dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)**</code>

  Linear Support Vector Classification

  ```
  >>> from sklearn.svm import LinearSVC
  >>> from sklearn.pipeline import make_pipeline
  >>> from sklearn.preprocessing import StandardScaler
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_features=4, random_state=0)
  >>> clf = make_pipeline(StandardScaler(),
  ...                     LinearSVC(random_state=0, tol=1e-5))
  >>> clf.fit(X, y)
  Pipeline(steps=[('standardscaler', StandardScaler()),
                  ('linearsvc', LinearSVC(random_state=0, tol=1e-05))])
  ```
  
- <code>**sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)**</code>

  ```
  >>> import numpy as np
  >>> from sklearn.pipeline import make_pipeline
  >>> from sklearn.preprocessing import StandardScaler
  >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
  >>> y = np.array([1, 1, 2, 2])
  >>> from sklearn.svm import SVC
  >>> clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
  >>> clf.fit(X, y)
  Pipeline(steps=[('standardscaler', StandardScaler()),
                  ('svc', SVC(gamma='auto'))])
  ```
  
