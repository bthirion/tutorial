"""
Test the searchlight module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import numpy as np

from nose.tools import assert_equal, assert_true

# Create a toy dataset to run searchlight on

# Initialize with 10x10x10 scans of random values on 30 frames
rand = np.random.RandomState(0)
frames = 30
data = rand.rand(frames, 10, 10, 10)

# Create a condition array
cond = np.array([int(i > (frames / 2)) for i in range(frames)])

# Create some activation pixels. Points are not connected.
mask = np.zeros((10, 10, 10), np.bool)
for i in [2, 4, 6]:
    for j in [3, 5, 7]:
        for k in [5, 7, 9]:
            mask[i, j, k] = True

# Apply a fake activation when condition is 1 (consider 0 as resting state)
data[cond][mask] = data[cond][mask] + 3

# Define score function
from sklearn.metrics import precision_score
score_func = precision_score

# Define cross validation
from sklearn.cross_validation import KFold
cv = KFold(cond.size, k=4)

from nisl import searchlight
n_jobs = 1

# The radius is the one of the Searchlight sphere that will scan the volume
# Small radius
sl = searchlight.SearchLight(mask, mask, radius=1.5,
                n_jobs=n_jobs, score_func=score_func, cv=cv)
sl.fit(data, cond)
assert_equal(np.nonzero(sl.scores_)[0].size, 6)

# Medium radius : Searchlight start to grab some points
sl = searchlight.SearchLight(mask, mask, radius=2.5,
                n_jobs=n_jobs, score_func=score_func, cv=cv)
sl.fit(data, cond)
assert_equal(np.nonzero(sl.scores_)[0].size, 22)

# Big radius (> sqrt(2) * 2), we get all activation points
sl = searchlight.SearchLight(mask, mask, radius=3,
                n_jobs=n_jobs, score_func=score_func, cv=cv)
sl.fit(data, cond)
assert_equal(np.nonzero(sl.scores_)[0].size, 27)

assert_true(sl.scores_[mask].all() > 0.1)
assert_true(sl.scores_[-mask].all() == 0)