import sys
import numpy as np


def logistic_link(x):
    return 1.0 / (1.0 + np.exp(-x))


def identity_link(x):
    return x


def load_dismec_style_predictions(file_name: str, link_function: callable):
    with open(file_name, "r") as file:
        _, k = map(int, next(file).split())
        scores = np.zeros((1000, k), dtype=np.float32)
        label_ids = np.zeros((1000, k), dtype=int)
        for i, line in enumerate(file):
            # potentially allocate more memory
            if i == scores.shape[0]:
                scores = np.concatenate((scores, np.zeros_like(scores)), axis=0)
                label_ids = np.concatenate((label_ids, np.zeros_like(label_ids)), axis=0)
            for j, pred in enumerate(line.split()):
                lbl, score = pred.split(":")
                scores[i, j] = float(score)
                label_ids[i, j] = int(lbl)

        scores = link_function(scores[:i+1, :])
        return scores, label_ids[:i+1, :]


def load_dismec_style_ground_truth(file_name: str):
    with open(file_name, "r") as file:
        num_instances, _, num_labels = map(int, next(file).split())
        labels_lol = []
        for i, line in enumerate(file):
            if line.startswith(" "):
                labels_lol.append([])
            else:
                labels_lol.append(set(map(int, line.split()[0].split(","))))
        return labels_lol, num_labels


def prepare_calibration_data(scores, label_ids, gt_ids, num_labels):
    ground_truth = np.zeros_like(scores, dtype=bool)
    label_coo = np.zeros((sum(map(len, gt_ids)), 2), dtype=int)
    counter = 0
    for i in range(scores.shape[0]):
        for k in range(scores.shape[1]):
            if label_ids[i, k] in gt_ids[i]:
                ground_truth[i, k] = 1
        for j in gt_ids[i]:
            label_coo[counter, 0] = i
            label_coo[counter, 1] = j
            counter += 1

    return ground_truth, label_coo


def convert_dismec_data(prediction_file: str, ground_truth_file: str, target_file: str, loss_fn: str):
    if loss_fn == "logistic":
        link = logistic_link
    elif loss_fn == "squared-hinge":
        link = identity_link
    scores, label_ids = load_dismec_style_predictions(prediction_file, link)
    gt_ids, num_labels = load_dismec_style_ground_truth(ground_truth_file)
    gt_scores, label_coo = prepare_calibration_data(scores, label_ids, gt_ids, num_labels)
    np.savez(target_file, scores=scores, prediction_ids=label_ids, 
        true_positives=gt_scores, labels=label_coo)


def main():
    convert_dismec_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
