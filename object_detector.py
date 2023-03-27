# ---------------------------------------------------------------------------------
# Object detection without training using self-supervised learning and transformers
# Written by Maria Culman
# Based on:
# DINO (https://arxiv.org/abs/2104.14294) 
# LOST (https://arxiv.org/abs/2109.14279)
# ConvNeXt (https://arxiv.org/abs/2201.03545)
# ---------------------------------------------------------------------------------

## Import generic libraries
import os
import json
import shutil
import tempfile

## Import specialized libraries
import numpy as np
from tqdm import tqdm
from PIL import Image
import skimage.io as io
import cv2
import requests
import gradio as gr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms
from scipy.ndimage import label, find_objects
import vision_transformer as vits

## Fuctions
def dino(arch, patch_size, device): 
    """
    Creation of a model (ViT) from DINO to extract features that help discover objects.
    Inputs
        arch (str): name of the Vision Transformer trained with DINO to be implemented
        patch_size (int): size of patches for the original to be devided. Options are 8 and 16 pixels. Smaller patches provided fine-grained discoveries.
        device (str): computer device where the model should be stored. CPU is preferred for inference.
    Outputs
        model (obj): model loaded with pre-trained weights
    """
    # Use auxiliary python file to create the model's achitecture
    model = vits.__dict__[arch](patch_size = patch_size, num_classes = 0)
    
    # Freeze the model so it is used as pre-trained
    for p in model.parameters():
        p.requires_grad = False

    # Initialize model with pre-trained weights on ImageNet-1k dataset
    if patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url)
    msg = model.load_state_dict(state_dict, strict=True)
    print("Pretrained weights found at {} and loaded with msg: {}".format(url, msg))

    # Make model avilable for inference in accessible device
    model.eval()
    model.to(device)
    return model

def lost(feats, dims, scales, init_image_size, n_seeds, correl_th):
    """
    Adaptation of LOST method to discover multiple objects.
    Inputs
        feats (tensor): the pixel/patche features of an image obtained from DINO
        dims (list of int): dimension of the map from which the features are used
        scales (list of int): from image to map scale
        init_image_size (int): size of the image
        n_seeds (int): number of potential seeds to discover objects from
        correl_th (int): correlation threshold to add pixels to the seed to be considered one object
    Outputs
        preds (list of arrays): object discoveries
        scores_scale (list of floats): correlation degree of object discoveries scaled to 0-1
    """
    # Compute the similarity
    A = (feats @ feats.transpose(1, 2)).squeeze()

    # Compute the inverse degree centrality measure per patch
    sorted_patches, scores = patch_scoring(A)

    # Select the initial seeds
    seeds = sorted_patches[:n_seeds]
    scores_seeds = scores[:n_seeds]
    
    # Cluster seeds together and select less correlating seed per cluster
    seeds_cluster, scores_cluster = clean_seeds(seeds, scores_seeds)
    
    # Scale correlating degree 0-1
    max_score = dims[0] * dims[1]
    scores_scale = (scores_cluster - (-max_score)) / (0 - (-max_score))
    
    # Create discovery box around seed considering the minimum correlation threshold 
    preds = []
    for s in seeds_cluster:
        # Box extraction
        pred, _ = detect_box(A[s, :], s, dims, correl_th, scales = scales, initial_im_size = init_image_size[1:], )
        preds.append(np.asarray(pred))

    return preds, scores_scale

def patch_scoring(M, threshold=0.):
    """
    Patch scoring based on the inverse degree.
    """
    # Cloning important
    A = M.clone()

    # Zero diagonal
    A.fill_diagonal_(0)

    # Make sure symmetric and non nul
    A[A < 0] = 0
    C = A + A.t()

    # Sort pixels by inverse degree
    cent = -torch.sum(A > threshold, dim=1).type(torch.float32)
    sort, sel = torch.sort(cent, descending=True)

    return sel, sort

def clean_seeds(seeds, scores):
    """
    Cluster spatially adjacent seeds and leave the seed with lowest correlation degree per cluster.
    """
    # Create auxiliary structures for clustering and scoring
    seed_array = np.zeros((w_featmap, h_featmap))
    scores_array = np.empty((w_featmap, h_featmap))
    scores_array[:] = np.NaN
    seed_array_2 = np.zeros((w_featmap, h_featmap))
    
    # Organize seeds per location 
    seeds_1, indices = torch.sort(seeds)
    scores_1 = scores[indices]
    
    # Locate seeds in clustering sructure
    for i, s in enumerate(seeds_1):
        center = np.unravel_index(s.cpu().numpy(), (w_featmap, h_featmap))
        seed_array[center[0]][center[1]] = 1
        scores_array[center[0]][center[1]] = scores_1[i].cpu().numpy()

    # Cluster adjacent seeds 
    labeled_seed, num_features = label(seed_array)

    # Leave one seed per cluster with lowest correlation degree
    seeds_2 = []
    for n in range(num_features):
        loc = find_objects(labeled_seed == (n + 1))[0]
        n_labeled_array = labeled_seed[loc]
        n_scores_array = scores_array[loc]

        ind = np.unravel_index(np.nanargmax(n_scores_array, axis=None), n_scores_array.shape)

        n_seeds_array_2 = np.zeros(n_labeled_array.shape)
        n_seeds_array_2[ind] = n + 1

        seed_array_2[loc] = n_seeds_array_2
        seeds_ind = np.asarray(np.where(seed_array_2 == n + 1)).T

        pos = np.ravel_multi_index(seeds_ind[0], (w_featmap, h_featmap))
        seeds_2.append(pos)

    # Resolve conflicting seeds in case of two with lowest degree
    seeds_3 = np.unique(np.array(seeds_2))
    seeds_final, indices = torch.sort(torch.as_tensor(seeds_3))
    c = np.isin(seeds_1.cpu().numpy(), seeds_final)
    scores_final = scores_1[c]

    return seeds_final, scores_final

def detect_box(A, seed, dims, correl_th, initial_im_size=None, scales=None):
    """
    Extract a box corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims

    correl = A.reshape(w_featmap, h_featmap).float()

    # Compute connected components
    labeled_array, num_features = label(correl.cpu().numpy() > correl_th)

    # Find connected component corresponding to the initial seed
    cc = labeled_array[np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))]
    if cc == 0:
        pred = [0, 0, 0, 0]
        pred_feats = [0, 0, 0, 0]

    else:
        # Find box
        mask = np.where(labeled_array == cc)
        # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1

        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax

        pred = [r_xmin, r_ymin, r_xmax, r_ymax]

        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])

        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

    return pred, pred_feats

def convnext(inp, final_pred):
    """
    Classify object discoveries. Among the predicted classes, select the one with highest confidence.
    Inputs
        inp (tensor and str): tensor of transformed image and image file path
        final_pred (list of arrays): object discoveries cleaned by non-maximum suppression
    Outputs
        labels (list of str): object classes from detections with highest confidence
        confidences (list of floats): object confidences from detections with highest confidence
    """
    
    # Open image 
    img_dis = Image.open(inp[1])
    
    # Create temporal folder to save cropped images
    tempdir = tempfile.mkdtemp()
    
    # Create cropped images centered at box discovery and use Hugging Face space of ConvNeXt to retrieve image classification
    labels = []
    confidences = []
    for i, p in enumerate(final_pred):
        [bbox_x, bbox_y, bbox_w, bbox_h] = p[0], p[1], p[2]-p[0], p[3]-p[1]

        # Crop image to standard size and save at temporary folder
        x_center = bbox_x + (bbox_w / 2)
        y_center = bbox_y + (bbox_h / 2)
        s = 128
        box_dis = img_dis.crop((x_center - (s / 2), y_center - (s / 2), x_center + (s / 2), y_center + (s / 2)))
        crop_path = os.path.join(tempdir, im_name + '_{}'.format(i) + '.jpg')
        box_dis.save(crop_path, "JPEG", quality=100)
        image = gr.processing_utils.encode_url_or_file_to_base64(crop_path)
        
        # Send image to ConvNeXt space and get classification results
        r = requests.post(url='https://hf.space/embed/akhaliq/convnext/+/api/predict/', json={"data": [image]})
        convnext = r.json().get('data')[0].get('confidences')
        
        # Keep label and confidence of classification with highest confidence.
        label = list(convnext[0].values())[0]
        labels.append(label)
        confidence = list(convnext[0].values())[1]
        confidences.append(confidence)
    
    # Eliminate temporary folder
    shutil.rmtree(tempdir)
    
    return labels, confidences

def visualize_predictions(image, preds, output_dir, im_name, number_seeds, correlation_th, labels, confidences):
    """
    Visualization of the predicted objects (boxes, classes, and confidences).
    """
    # Define colors for visualization
    COLORS = [[0.000, 0.500, 0.850], [0.750, 0.425, 0.100], [0.950, 0.700, 0.150],
          [0.500, 0.200, 0.600], [0.400, 0.650, 0.150], [0.300, 0.800, 0.950]]
    colors = COLORS * 100
   
    # Open image in a plot
    plt.imshow(image)
    ax = plt.gca()
    
    # Draw each prediction if its confidence is above 0.25    
    for (xmin, ymin, xmax, ymax), label, conf, c in zip(preds, labels, confidences, colors):
        label = label[:label.find(",")] if label.find(',') != -1 else label
        if conf > 0.25:
            w = xmax - xmin
            h = ymax - ymin
            ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
                                       fill = False, color = c, linewidth = 3))
            text = f'{label}: {conf:.2f}'
            ax.text(xmin, ymin, text, fontsize = 15,
                    bbox = dict(facecolor = 'yellow', alpha = 0.5))
        else:
            continue
    
    # Remove axis from plot, and save and display plot
    plt.axis('off')
    pltname = f"{output_dir}/pred_{im_name}_{number_seeds}_{correlation_th}.jpg"
    plt.savefig(pltname)
    plt.show()

## Classes 
class DatasetCustom:
    """
    Class to instantiate a custom dataset that opens and handles images as necessary.
    Images contained in the image folder are loaded as tensors.
    Fuctions to retrieve the image name and load it are made available.
    """    
    def __init__(self, image_folder):
        self.image_dir = image_folder
        images = [os.path.join(r, fn) for r, ds, fs in os.walk(self.image_dir) for fn in fs if fn.endswith('.jpg')]
        images.sort()
        
        transform = pth_transforms.Compose(
            [pth_transforms.ToTensor(), 
             pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])
        
        # Load tensor of transformed image and image file path 
        self.dataloader = [[transform((Image.open(im)).convert("RGB")), im] for im in images]

    def get_image_name(self, inp):
        """
        Return the image name
        """
        file_name = os.path.basename(inp)
        im_name = "{}".format(file_name[:file_name.rfind(".")])

        return im_name

    def load_image(self, im_name):
        """
        Load the image corresponding to the im_name
        """
        im = os.path.join(self.image_dir, im_name + '.jpg')
        image = io.imread(im)
        return image

## Main code
if __name__ == "__main__":
    
    # -------------------------------------------------------------------------------------------------------
    # Declare folder paths for input images and output results
    image_dir = "./images/"
    output_dir = "./output/"
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------------------------------------
    # Define object discovery parameters
    patch_size = 8 # Options are 8 and 16 pixels
    number_seeds = 40 # Number of potential objects to detect
    correlation_th = 75 # Minimum correlation to extend an object discovery 

    # -------------------------------------------------------------------------------------------------------
    # Dataset
    dataset = DatasetCustom(image_dir) # Creates dataset object

    # -------------------------------------------------------------------------------------------------------
    # Device and Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # Defines place to load model
    model = dino("vit_base", patch_size, device) # Creates and loads model from DINO

    # -------------------------------------------------------------------------------------------------------
    # Loop over images to discover and classify objects
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING ---------------------------------------------------------------------
        img = inp[0]
        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / patch_size) * patch_size),
            int(np.ceil(img.shape[2] / patch_size) * patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # Move to gpu
        # img = img.cuda(non_blocking=True) # Make line available if working in GPU device instead of CPU device
        
        # Size for transformers
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size


        # ------------ 1. Apply DINO to extract features -------------------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS ---------------------------------------------------------------------
            # Store the outputs of qkv layer from the last attention layer
            feat_out = {}

            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output

            model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

            # Forward pass in the model
            attentions = model.get_last_selfattention(img[None, :, :, :])

            # Scaling factor
            scales = [patch_size, patch_size]

            # Dimensions
            nb_im = attentions.shape[0]  # Batch size
            nh = attentions.shape[1]  # Number of heads
            nb_tokens = attentions.shape[2]  # Number of tokens

            # Extract the qkv features of the last attention layer
            qkv = (feat_out["qkv"]
                    .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                    .permute(2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

            # Select the keys extracted from the layer
            feats = k[:, 1:, :]

        # ------------ 2. Apply LOST for multiple-object discovery ---------------------------------------------
        # Discover the potential objects located in the image
        preds, scores = lost(
            feats,
            [w_featmap, h_featmap],
            scales,
            init_image_size,
            number_seeds, correlation_th)

        # Apply non-maximum suppression to eliminate overlapping discoveries
        indices = cv2.dnn.NMSBoxes(preds, scores.cpu().numpy(), score_threshold=0.4, nms_threshold=0.5)
        final_pred = np.array(preds)[indices]
        final_scores = np.array(scores.cpu().numpy())[indices]
        
        # ------------ 3. Apply ConvNeXt to classify object discoveries ---------------------------------------- 
        # Assign the most probable class to each of the object discoveries
        labels, confidences = convnext(inp, final_pred)
        print(f'\nPredictions for image {im_name}:')
        mapped = zip(labels, confidences)
        print(set(mapped))

        # ------------ 4. Visualize image with detection results -----------------------------------------------
        image = dataset.load_image(im_name)
        visualize_predictions(image, final_pred, output_dir, im_name, number_seeds, correlation_th, labels, confidences)
