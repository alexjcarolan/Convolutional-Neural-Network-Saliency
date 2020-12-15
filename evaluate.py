import math
import pickle
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage

parser = argparse.ArgumentParser(description="Evaluating model outputs")
parser.add_argument("--preds", default="preds.pkl", type=str, help="Path to prediction set")
parser.add_argument("--vals", default="vals.pkl", type=str, help="Path to validation set")

def evaluate(preds, vals):
    preds = pickle.load(open(preds, "rb"))
    vals = pickle.load(open(vals, "rb"))

    cc_scores = []
    auc_borji_scores = []
    auc_shuffled_scores = []
    for i in range(len(preds)):
        val = vals[i]['y_original']
        pred = np.reshape(preds[i], (48, 48))
        pred = Image.fromarray((pred * 255).astype(np.uint8)).resize((val.shape[1], val.shape[0]))
        pred = np.asarray(pred, dtype='float32') / 255.
        pred = ndimage.gaussian_filter(pred, sigma=2)
        cc_scores.append(cc(pred, val))
        auc_borji_scores.append(auc_borji(pred, np.asarray(val, dtype=np.int)))

        other = np.zeros((val.shape[0], val.shape[1]), dtype=np.int)
        randind_maps = np.random.choice(len(vals), size=10, replace=False)
        for i in range(10):
            other = other | np.asarray(vals[randind_maps[i]]['y_original'], dtype=np.int)

        auc_shuffled_scores.append(auc_shuff(pred, np.asarray(val, dtype=np.int), other))

    return np.mean(cc_scores), np.mean(auc_shuffled_scores), np.mean(auc_borji_scores)

def normalize_map(s_map):
    norm_s_map = (s_map - np.min(s_map)) / (np.max(s_map) - np.min(s_map))
    return norm_s_map

def auc_borji(s_map, val, splits=100, stepsize=0.1):
    s_map = normalize_map(s_map)

    S = s_map.flatten()
    F = val.flatten()

    Sth = S[F > 0]
    num_fixations = Sth.shape[0]
    num_pixels = S.shape[0]

    r = np.random.randint(num_pixels, size=(splits, num_fixations))
    randfix = np.zeros((splits, num_fixations))

    for i in range(splits):
        randfix[i, :] = S[r[i, :]]

    aucs = []
    for i in range(splits):
        curfix = randfix[i, :]

        allthreshes = np.arange(0, np.max(np.concatenate((Sth, curfix), axis=0)), stepsize)
        allthreshes = allthreshes[::-1]

        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1.0
        fp[-1] = 1.0

        tp[1:-1] = [float(np.sum(Sth >= thresh)) / num_fixations for thresh in allthreshes]
        fp[1:-1] = [float(np.sum(curfix >= thresh)) / num_fixations for thresh in allthreshes]

        aucs.append(np.trapz(tp, fp))

    return np.mean(aucs)

def auc_shuff(s_map, val, other_map, splits=100, stepsize=0.1):
    s_map = normalize_map(s_map)

    S = s_map.flatten()
    F = val.flatten()
    oth = other_map.flatten()

    Sth = S[F > 0]
    num_fixations = Sth.shape[0]

    ind = np.flatnonzero(oth)

    num_fixations_others = min(ind.shape[0], num_fixations)
    randfix = np.zeros((splits, num_fixations_others))

    for i in range(splits):
        randind = ind[np.random.permutation(ind.shape[0])]
        randfix[i, :] = S[randind[:num_fixations_others]]

    aucs = []
    for i in range(splits):
        curfix = randfix[i, :]

        allthreshes = np.arange(0, np.max(np.concatenate((Sth, curfix), axis=0)), stepsize)
        allthreshes = allthreshes[::-1]

        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1.0
        fp[-1] = 1.0

        tp[1:-1] = [float(np.sum(Sth >= thresh)) / num_fixations for thresh in allthreshes]
        fp[1:-1] = [float(np.sum(curfix >= thresh)) / num_fixations_others for thresh in allthreshes]

        aucs.append(np.trapz(tp, fp))

    return np.mean(aucs)

def cc(s_map, val):
    sigma = 19
    val = ndimage.gaussian_filter(val, sigma)
    val = val - np.min(val)
    val = val / np.max(val)

    s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)
    val_norm = (val - np.mean(val)) / np.std(val)
    a = s_map_norm
    b = val_norm
    r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
    return r

if __name__ == "__main__":
    args = parser.parse_args()
    cc, auc_borji, auc_shuffled = evaluate(args.preds, args.vals)
    print(f"CC: {cc} AUC Borji {auc_borji} AUC Shuffled {auc_shuffled}")