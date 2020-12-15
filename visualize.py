import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import ndimage
from matplotlib.pyplot import imshow

parser = argparse.ArgumentParser(description="Visualizing model outputs")
parser.add_argument("--preds", default="preds.pkl", type=str, help="Path to prediction set")
parser.add_argument("--vals", default="vals.pkl", type=str, help="Path to validation set")
parser.add_argument("--out-directory", default=Path("outputs"), type=Path, help="Output directory path")

def visualize(preds, vals, out_directory):
    # loading preds and vals
    preds = pickle.load(open(preds, "rb"))
    vals = pickle.load(open(vals, "rb"))

    index = np.random.randint(0, len(preds), size=3) # get indices for 3 random images

    outputs = []
    for idx in index:
        # getting original image
        image = vals[idx]['X_original']
        image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
        outputs.append(image)

        # getting ground truth saliency map
        sal_map = vals[idx]['y_original']
        sal_map = ndimage.gaussian_filter(sal_map, 19)
        outputs.append(sal_map)

        # getting model prediction
        pred = np.reshape(preds[idx], (48, 48))
        pred = Image.fromarray((pred * 255).astype(np.uint8)).resize((image.shape[1], image.shape[0]))
        pred = np.asarray(pred, dtype='float32') / 255.
        pred = ndimage.gaussian_filter(pred, sigma=2)
        outputs.append(pred)

    # plotting images 
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(32,32))
    ax[0][0].set_title("Image", fontsize=40)
    ax[0][1].set_title("Validation", fontsize=40)
    ax[0][2].set_title("Prediction", fontsize=40)
    
    fig.tight_layout()

    for i, axi in enumerate(ax.flat):
        axi.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axi.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        axi.imshow(outputs[i])
    
    # saving output
    outpath = os.path.join(out_directory, "visualization.png")
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0)
    
if __name__ == "__main__":
    args = parser.parse_args()
    visualize(args.preds, args.vals, args.out_directory)