# Means-Testing
A simple library to compare the means of two or more *independent* samples and check for statistical differences.

## Installation
```
pip install means-testing
```

## Basic usage
```python
from means_testing import MeansTester

# Initialize a MeansTester, that takes in samples and tests for statistical differences
StatToolbox = MeansTester(*samples)

# Check if means are significantly different
StatToolbox.test_means()
```

After carrying out the statistical test, one can determine if the samples passed are statistically different or not via the property `SignificantDifference` as shown below.
```python
StatToolbox.SignificantDifference
> True (if samples are statistically different) or
> False (if samples are NOT statistically different)
```

For more examples check out the [examples](https://github.com/RenZhen95/means-testing/tree/master/examples).

## Citation
The functions and statistical tests here are impelemented based on the book by Riffenburgh, *Statistics in Medicine*:


Riffenburgh R H. Statistics in Medicine. Elsevier/Academic Press, Amsterdam, Netherlands; 2012
