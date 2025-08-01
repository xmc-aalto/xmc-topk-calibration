import numpy as np
from collections import defaultdict
import json
import sys
from pathlib import Path
from torch import tensor
from torchmetrics.classification import BinaryCalibrationError
import matplotlib.pyplot as plt
from sklearn import calibration, isotonic
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import brier_score_loss
import tikzplotlib
import torch
from torch import nn, optim
from torch.nn import functional as F
from typing import Optional, Tuple, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.calibration_error import _binning_bucketize
from torchmetrics.utilities.data import dim_zero_cat
from calibration_metrics import ace
import warnings
warnings.filterwarnings("ignore")


def _ce_compute_modified(
    confidences: Tensor,
    accuracies: Tensor,
    bin_boundaries: Union[Tensor, int],
    mtype: str = "all",
    norm: str = "l1",
    debias: bool = False,) -> Tensor:
    """Compute the calibration error given the provided bin boundaries and norm.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.
        norm: Norm function to use when computing calibration error. Defaults to "l1".
        debias: Apply debiasing to L2 norm computation as in
            `Verified Uncertainty Calibration`_. Defaults to False.

    Raises:
        ValueError: If an unsupported norm function is provided.

    Returns:
        Tensor: Calibration error scalar.

    """
    if isinstance(bin_boundaries, int):
        bin_boundaries = torch.linspace(0, 1, bin_boundaries + 1, dtype=confidences.dtype, device=confidences.device)

    if norm not in {"l1", "l2", "max"}:
        raise ValueError(f"Argument `norm` is expected to be one of 'l1', 'l2', 'max' but got {norm}")

    with torch.no_grad():
        acc_bin, conf_bin, prop_bin = _binning_bucketize(confidences, accuracies, bin_boundaries)

    #different types of caliibration error, ece-all, ece-over, ece-under
    
    if mtype == "all":
        if norm == "l1":
            return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
        if norm == "max":
            ce = torch.max(torch.abs(acc_bin - conf_bin))
        if norm == "l2":
            ce = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
            # NOTE: debiasing is disabled in the wrapper functions. This implementation differs from that in sklearn.
            if debias:
                # the order here (acc_bin - 1 ) vs (1 - acc_bin) is flipped from
                # the equation in Verified Uncertainty Prediction (Kumar et al 2019)/
                debias_bins = (acc_bin * (acc_bin - 1) * prop_bin) / (prop_bin * accuracies.size()[0] - 1)
                ce += torch.sum(torch.nan_to_num(debias_bins))  # replace nans with zeros if nothing appeared in a bin
            return torch.sqrt(ce) if ce > 0 else torch.tensor(0)
    elif mtype == "over":
        if norm == "l1":
            return torch.sum(torch.max(conf_bin-acc_bin,torch.tensor([0])) * prop_bin)
        if norm == "max":
            ce = torch.max(torch.max(conf_bin-acc_bin,torch.tensor([0])))
        if norm == "l2":
            ce = torch.sum(torch.pow(torch.max(conf_bin-acc_bin,torch.tensor([0])),2)*prop_bin)
            return torch.sqrt(ce) if ce > 0 else torch.tensor(0)
    
    elif mtype == "under":
        if norm == "l1":
            return torch.sum(torch.max(acc_bin - conf_bin,torch.tensor([0])) * prop_bin)
        if norm == "max":
            ce = torch.max(torch.max(acc_bin - conf_bin,torch.tensor([0])))
        if norm == "l2":
            ce = torch.sum(torch.pow(torch.max(acc_bin - conf_bin,torch.tensor([0])),2)*prop_bin)
            return torch.sqrt(ce) if ce > 0 else torch.tensor(0)
        
    return ce

class ModifierBinaryCalibrationError(BinaryCalibrationError):
    def __init__(self,n_bins: int = 15,norm: Literal["l1", "l2", "max"] = "l1",mtype="all"):
        super().__init__(n_bins,norm)
        self.mtype = mtype
        
    def compute(self) -> Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute_modified(confidences, accuracies, self.n_bins,mtype=self.mtype,norm=self.norm)
        
    



def make_calibration_curve(scores: np.ndarray, ground_truth: np.ndarray, n_bins: int = 15):
    prob_true, prob_pred = calibration.calibration_curve(
        ground_truth.flatten(),
        scores.flatten(),
        n_bins=n_bins)

    return {
        "prob-true": [round(x, 4) for x in prob_true],
        "prob-pred": [round(x, 4) for x in prob_pred],
    }


def calc_ece(scores, ground_truth, n_bins: int):
    ece_metric = BinaryCalibrationError(n_bins=n_bins, norm='l1')
    return ece_metric(tensor(scores.flatten()), tensor(ground_truth.flatten())).item()

def calc_ece_modified(scores, ground_truth, n_bins: int):
    mtypes = ["all","over","under"]
    ece_values = []
    for mtype in mtypes:
        ece_metric = ModifierBinaryCalibrationError(n_bins=n_bins, norm='l1',mtype=mtype)
        ece_values.append(ece_metric(tensor(scores.flatten()), tensor(ground_truth.flatten())).item())
    return ece_values

def calc_nlll(scores,ground_truth):
    nlll_criterion = nn.BCELoss()
    nlll_score = nlll_criterion(torch.tensor(scores).double(), torch.tensor(ground_truth).double()).item()
    #print(nlll_score)
    return nlll_score

def evaluate(scores: np.ndarray, ground_truth: np.ndarray, total: int, n_bins: int = 15):
    result_dict = defaultdict(dict)

    #for k in (1, 3, 5, 10,50,100,3993):
    for k in (1, 3, 5, 10, 25, 50, 100):
        scores_at_k = scores[:, :k]
        ece = calc_ece(scores_at_k, ground_truth[:, :k], n_bins=n_bins)
        ece_values = calc_ece_modified(scores_at_k, ground_truth[:, :k], n_bins=n_bins)
        p_k = np.mean(ground_truth[:, :k])
        r_k = (np.sum(ground_truth[:, :k])) / total
        result_dict[k]["ECE"] = round(float(ece), 4)
        result_dict[k]["precision"] = round(float(p_k), 4)
        result_dict[k]["micro-recall"] = round(float(r_k), 4)
        result_dict[k]["ECE-all"] = round(float(ece_values[0]), 4)
        result_dict[k]["ECE-over"] = round(float(ece_values[1]), 4)
        result_dict[k]["ECE-under"] = round(float(ece_values[2]), 4)
        result_dict[k]['Brier'] = round(float(brier_score_loss(ground_truth[:, :k].flatten(), scores_at_k.flatten())),4)
        result_dict[k]["ace"] = round(float(ace(scores_at_k.flatten(), ground_truth[:, :k].flatten(), n_bin=n_bins, mode='l1')),4)
        result_dict[k]['nlll'] = round(float(calc_nlll(scores_at_k.flatten(),ground_truth[:, :k].flatten())),4)

        # calibration curve
        result_dict[k]["calibration-curve"] = make_calibration_curve(scores_at_k, ground_truth[:, :k], n_bins=n_bins)

        # histogram
        hist, _ = np.histogram(scores_at_k.flatten(), bins=np.linspace(0, 1, n_bins + 1, endpoint=True))
        result_dict[k]["histogram"] = [int(x) for x in hist]

        # isotonic regression
        calibrated_scores = run_isotonic_regression(scores_at_k, ground_truth[:, :k])
        ece_c = calc_ece(calibrated_scores, ground_truth[:, :k], n_bins=n_bins)
        ece_values_c = calc_ece_modified(calibrated_scores, ground_truth[:, :k], n_bins=n_bins)
        result_dict[k]["ECE-I"] = round(float(ece_c), 4)
        result_dict[k]["ECE-I-all"] = round(float(ece_values_c[0]), 4)
        result_dict[k]["ECE-I-over"] = round(float(ece_values_c[1]), 4)
        result_dict[k]["ECE-I-under"] = round(float(ece_values_c[2]), 4)
        result_dict[k]['Brier-I'] = round(float(brier_score_loss(ground_truth[:, :k].flatten(), calibrated_scores.flatten())),4)
        result_dict[k]["ace-I"] = round(float(ace(calibrated_scores.flatten(), ground_truth[:, :k].flatten(), n_bin=n_bins, mode='l1')),4)
        result_dict[k]["nlll-I"] = round(float(calc_nlll(calibrated_scores.flatten(),ground_truth[:, :k].flatten())),4)
        
        
        # platt scaling
        calibrated_scores_p = run_platt_scaling(scores_at_k, ground_truth[:, :k])
        ece_c_p = calc_ece(calibrated_scores_p, ground_truth[:, :k], n_bins=n_bins)
        ece_values_cp = calc_ece_modified(calibrated_scores_p, ground_truth[:, :k], n_bins=n_bins)
        result_dict[k]["ECE-P"] = round(float(ece_c_p), 4)
        result_dict[k]["ECE-P-all"] = round(float(ece_values_cp[0]), 4)
        result_dict[k]["ECE-P-over"] = round(float(ece_values_cp[1]), 4)
        result_dict[k]["ECE-P-under"] = round(float(ece_values_cp[2]), 4)
        result_dict[k]['Brier-P'] = round(float(brier_score_loss(ground_truth[:, :k].flatten(), calibrated_scores_p.flatten())),4)
        result_dict[k]["ace-P"] = round(float(ace(calibrated_scores_p.flatten(), ground_truth[:, :k].flatten(), n_bin=n_bins, mode='l1')),4)
        result_dict[k]["nlll-P"] = round(float(calc_nlll(calibrated_scores_p.flatten(),ground_truth[:, :k].flatten())),4)

        # new calibration curve
        result_dict[k]["isotonic"] = make_calibration_curve(calibrated_scores, ground_truth[:, :k], n_bins=n_bins)
        
        #new histogram
        hist, _ = np.histogram(calibrated_scores.flatten(), bins=np.linspace(0, 1, n_bins + 1, endpoint=True))
        result_dict[k]["histogram-I"] = [int(x) for x in hist]

    # full recalibration
    calibrated_scores = run_isotonic_regression(scores, ground_truth)


    # re-calibrate to re-rank
    re_rank = np.zeros_like(scores[:, :10])
    for k in range(10):
        re_rank[:, k:k+1] = run_isotonic_regression(scores[:, k:k+1], ground_truth[:, k:k+1])
        #re_rank[:, k:k+1] = run_platt_scaling(scores[:, k:k+1], ground_truth[:, k:k+1])
    # now, actually do the re-sorting
    sort_idx = np.argsort(-re_rank, axis=1)
    re_ranked_gt = np.take_along_axis(ground_truth, sort_idx, axis=1)
    p_at_1 = np.mean(re_ranked_gt[:, 0])
    p_at_3 = np.mean(re_ranked_gt[:, :3])
    p_at_5 = np.mean(re_ranked_gt[:, :5])
    print(f"P@1: {100*np.mean(ground_truth[:, 0]):.2f} -> {100*p_at_1:.2f}")
    print(f"P@3: {100*np.mean(ground_truth[:, :3]):.2f} -> {100*p_at_3:.2f}")
    print(f"P@5: {100*np.mean(ground_truth[:, :5]):.2f} -> {100*p_at_5:.2f}")

    return dict(result_dict), calibrated_scores


def run_isotonic_regression(scores: np.ndarray, ground_truth: np.ndarray):
    parts = np.linspace(0, scores.shape[0], 11, endpoint=True).astype(int)
    result = np.zeros_like(scores)
    regressor = isotonic.IsotonicRegression(out_of_bounds="clip")

    def regress(train_scores, train_targets, test_scores):
        regressor.fit(train_scores.flatten(), train_targets.flatten())
        return regressor.transform(test_scores.flatten()).reshape(test_scores.shape)

    for s, e in zip(parts[:-1], parts[1:]):
        src_scores = np.concatenate([scores[:s, :], scores[e:, :]], axis=0)
        src_truth = np.concatenate([ground_truth[:s, :], ground_truth[e:, :]], axis=0)
        fit_scores = regress(src_scores, src_truth, scores[s:e, :])
        fit_truth = ground_truth[s:e, :]
        result[s:e, :] = fit_scores #probably averaging

    return result



def run_platt_scaling(scores: np.ndarray, ground_truth: np.ndarray):
    parts = np.linspace(0, scores.shape[0], 11, endpoint=True).astype(int)
    result = np.zeros_like(scores)

    def logistic(train_scores, train_targets, test_scores):
        lr = LR()
        lr.fit( train_scores.flatten().reshape(-1,1), train_targets.flatten().reshape(-1,1))
        p_calibrated = lr.predict_proba( test_scores.flatten().reshape(-1,1))[:,1]
        return p_calibrated.reshape(test_scores.shape)
        
    for s, e in zip(parts[:-1], parts[1:]):
        src_scores = np.concatenate([scores[:s, :], scores[e:, :]], axis=0)
        src_truth = np.concatenate([ground_truth[:s, :], ground_truth[e:, :]], axis=0)
        fit_scores = logistic(src_scores, src_truth, scores[s:e, :])
        result[s:e, :] = fit_scores

    return result

def print_results(data: dict):
    for k, vals in data.items():
        ece = vals["ECE"]
        p_k = vals["precision"]
        r_k = vals["micro-recall"]
        print(f"ECE@{k:<3}: {100 * ece:6.2f}%")
        print(f"  P@{k:<3}: {100 * p_k:6.2f}%")
        print(f"miR@{k:<3}: {100 * r_k:6.2f}%")
        print(f"ECE@{k:<3}: {100 * vals['ECE-I']:6.2f}% (isotonic)")
        print(f"ECE@{k:<3}: {100 * vals['ECE-P']:6.2f}% (platt)")
        print()


def plot_calibration_curves(scores, ground_truth):
    from sklearn.calibration import CalibrationDisplay
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid()
    print(f"Maximum tail probability: {np.max(scores[:, -1]) * 100:.2f}%")
    for k in (1,3,5,25,50,100):
        disp = CalibrationDisplay.from_predictions(ground_truth[:, :k].flatten(), scores[:, :k].flatten(), ax=ax, name=f"@{k}")
    return fig


def main():
    link=None
    source = sys.argv[1]
    model = sys.argv[2]
    dataset = sys.argv[3]
    if len(sys.argv) > 4:
        link = sys.argv[4]
    target = f"../data_kddR/{model.lower()}_{dataset.lower()}.json"
    #target = f"../data_bin_xp/{model.lower()}_{dataset.lower()}_bin10.json"
    #target = f"../meta_experiment/{model.lower()}_{dataset.lower()}.json"

    data = np.load(source)
    scores = data['scores']
    if link == "range":
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    elif link == "margin":
        scores = np.clip(scores, -1, 1) * 0.5 + 0.5
    ground_truth = data['true_positives']
    total = len(data['labels'])

    result, scores_new = evaluate(scores, ground_truth, total,n_bins=10)
    print_results(result)

    t = Path(source)
    t = t.with_stem(t.stem + "-calib")
    np.savez(t, scores=scores_new, prediction_ids=data['prediction_ids'],
             true_positives=data['true_positives'], labels=data['labels'])

    Path(target).write_text(json.dumps({
        "model": model,
        "dataset": dataset,
        "results": result}, 
        indent=2)
    )

    fig = plot_calibration_curves(scores, ground_truth)
    #tikzplotlib.save(f"../figures/{model.lower()}_{dataset.lower()}.tex")
    #plt.savefig(f"../figures_png/{dataset.lower()}/{model.lower()}_{dataset.lower()}.png")
    

if __name__ == "__main__":
    main()
