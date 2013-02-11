"""
The haxby dataset: face vs house in object recognition
=======================================================

A significant part of the running time of this example is actually spent
in loading the data: we load all the data but only use the face and
houses conditions.
"""

from nisl import datasets
import numpy as np
from nibabel import load

from sklearn.cluster import Ward
from sklearn.feature_extraction.image import grid_to_graph

from parietal.probabilistic_parcellation import wardlet


### Load Haxby dataset ########################################################

dataset_files = datasets.fetch_haxby_simple()

# fmri_data and mask are copied to lose the reference to the original data
bold_img = load(dataset_files.func)
fmri_data = np.copy(bold_img.get_data())
affine = bold_img.get_affine()
y, session = np.loadtxt(dataset_files.session_target).astype("int").T
conditions = np.recfromtxt(dataset_files.conditions_target)['f0']
mask = dataset_files.mask

### Preprocess data ###########################################################
# Build the mean image because we have no anatomic data
mean_img = fmri_data.mean(axis=-1)

### Restrict to faces and houses ##############################################

# Keep only data corresponding to face or houses
condition_mask = (conditions != 'rest')
X = fmri_data[..., condition_mask]
y = y[condition_mask]
session = session[condition_mask]
conditions = conditions[condition_mask]

# We have 2 conditions
n_conditions = np.size(np.unique(y))

### Loading step ##############################################################
from nisl.io import NiftiMasker
from nibabel import Nifti1Image
nifti_masker = NiftiMasker(mask=mask, sessions=session, smooth=False, 
                           detrend=True, memory="nisl_cache", memory_level=1)
niimg = Nifti1Image(X, affine)
X = nifti_masker.fit_transform(niimg)

mask = load(nifti_masker.mask).get_data()
shape = load(nifti_masker.mask).shape

connectivity = grid_to_graph(*shape, mask=mask).tocsr()
#ward = Ward(n_clusters=1, connectivity=connectivity).fit(X.T)
#diff_data, _ = wardlet.wardlet_decomposition(ward, X.T)
#np.sqrt((diff_data ** 2).sum(1) / X.shape[1])

ward = Ward(n_clusters=10000, connectivity=connectivity).fit(X.T)
labels = ward.labels_
X_ = np.array([X[:, labels == l].mean(1) for  l  in np.unique(labels)]).T

### Prediction function #######################################################

### Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel and C=1
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1.)

### Dimension reduction #######################################################

from sklearn.feature_selection import SelectKBest, f_classif

### Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. We set the number of features to be selected to 1000
feature_selection = SelectKBest(f_classif, k=150)

# We have our classifier (SVC), our feature selection (SelectKBest), and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])

### Cross validation ##########################################################

from sklearn.cross_validation import LeaveOneLabelOut

### Define the cross-validation scheme used for validation.
# Here we use a LeaveOneLabelOut cross-validation on the session label
# divided by 2, which corresponds to a leave-two-session-out
cv = LeaveOneLabelOut(session)

### Compute the prediction accuracy for the different folds (i.e. session)
cv_scores = []
for train, test in cv:
    y_pred = anova_svc.fit(X_[train], y[train]) \
        .predict(X_[test])
    cv_scores.append(np.sum(y_pred == y[test]) / float(np.size(y[test])))

### Print results #############################################################

### Return the corresponding mean prediction accuracy
classification_accuracy = np.mean(cv_scores)

### Printing the results
print "=== ANOVA ==="
print "Classification accuracy: %f" % classification_accuracy, \
    " / Chance level: %f" % (1. / n_conditions)
