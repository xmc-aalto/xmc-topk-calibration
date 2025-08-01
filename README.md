# xmc-topk-calibration
Official Code for the paper How Well Calibrated are Extreme Multi-label Classifiers? An Empirical Analysis
---
In order to have a unified evaluation framework for the different XMLC methods,
we use the following intermediate format:

The result of the evaluation of a model on a dataset is stored as one `.npz` 
file, that contains four arrays:
* `scores`: scores of the top-k labels for each instance, sorted in descending order.
Of shape `num_instances x k`
* `true_positives`: boolean array of shape `num_instances x k` that contains `True`
if the corresponding label is relevant
* `prediction_ids`: integer array of shape `num_instances x k` that contains the corresponding
label ids of the predicted labels
* `labels`: integer array of shape `num_true_labels x 2`, with the first coordinate indicating the instance id, and the
second the label id. This corresponds to storing the labels as a COO matrix, skipping the values part, as all of them would
be one.

e.g For `DiSMEC` models, the `dismec_convert.py` script handles the conversion.

These npz files can then be fed into the pipeline that does the calibration evaluation, independently of
the underlying model. To that end, run `evaluate_new.py $NPZ_FILE $MODEL_NAME $DATASET_NAME`. This will
produce a small `.json` file in `../data/` that contain the actual evaluation.

