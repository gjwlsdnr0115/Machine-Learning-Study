# Week 2
`#End-to-End` `#Classification`

##

### Chapters
- Chapter 4: End-to_End Machine Learning Project
- Chapter 5: Classification 

##

### Practice Codes
- [CH04](./codes/CH04_training_models.ipynb)
- [CH05](./codes/CH05_support_vector_machines.ipynb)

##

### Takeaways
- <code>**SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, rho=0.85, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, p=0.1, seed=0, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False)**</code>

  **loss** = The loss function to be used. Defaults to ‘squared_loss’ which refers to the ordinary least squares fit. ‘huber’ is an epsilon insensitive loss function for robust regression.
  **penalty** = str, ‘l2’ or ‘l1’ or ‘elasticnet’
  **alpha** = Constant that multiplies the regularization term. Defaults to 0.0001
  **rho** = The Elastic Net mixing parameter, with 0 < rho <= 1. Defaults to 0.85.
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

  
