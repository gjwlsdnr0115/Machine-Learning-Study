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
