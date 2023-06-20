import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict as ddict
import matplotlib as mpl
from torch.utils.data import TensorDataset, DataLoader,Dataset, random_split
import json
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torchvision import models
from XAI.Integrated_gradient.implement.ig_image_pt import *
from XAI.guided_integrated_gradients.Implement.gig import *
from vision.utils.data import *
from vision.utils.model import *
from vision.utils.vision import *

from matplotlib import pylab as P
import numpy as np
import PIL.Image

import torch
from torchvision import models, transforms

# From our repository.
import saliency.core as saliency

# Boilerplate methods.
def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)

def ShowHeatMap(im, title, ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='inferno')
    P.title(title)

def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = im.resize((32, 32))
    im = np.asarray(im)
    return im

transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def PreprocessImages(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    images = transformer.forward(images)
    return images.requires_grad_(True)


im_orig = LoadImage('./doberman.png')
im_tensor = PreprocessImages([im_orig])
# Show the image
ShowImage(im_orig)

model=ResNet18(3)
path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
model.load_state_dict(torch.load(path+'/Distillation/assets/model/resnet18-baseline-1.pt'))
model.eval()
class_idx_str = 'class_idx_str'
predictions = model(im_tensor)
predictions = predictions.detach().numpy()
prediction_class = np.argmax(predictions[0])
call_model_args = {class_idx_str: prediction_class}

print("Prediction class: " + str(prediction_class))  # Should be a doberman, class idx = 236
im = im_orig.astype(np.float32)

## Gig test

test_set=experiment_cifar10(path+'/Distillation/assets/data/cifar10/experiment',train=False)

test_loader=DataLoader(test_set,batch_size=128,shuffle=False)

for j in test_loader:
    test_img=j[0][:,:3]
    test_ig=j[0][:,3:]
    test_label=j[1]
    break

test_img.shape

baseline=torch.zeros(1,3,32,32).to('cuda:1')

model.to('cuda:1')

mine=guided_ig(im_tensor.to('cuda:1'), baseline, torch.LongTensor([5]).to('cuda:1') ,model, steps=128, fraction=0.1, max_dist=0.02)

plt.imshow(mine[0].to('cpu').detach().numpy().transpose(1,2,0))


def call_model_function(images, call_model_args=None, expected_keys=None):
    images = PreprocessImages(images)
    target_class_idx =  call_model_args[class_idx_str]
    output = model(images)
    m = torch.nn.Softmax(dim=1)
    output = m(output)

    outputs = output[:,target_class_idx]
    grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
    grads = torch.movedim(grads[0], 1, 3)
    gradients = grads.detach().numpy()

    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}



# Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
integrated_gradients2 = saliency.IntegratedGradients()
guided_ig2 = GuidedIGnp()

# Baseline is a black image for vanilla integrated gradients.
baseline = np.zeros(im.shape)



# Compute the vanilla mask and the Guided IG mask.
vanilla_integrated_gradients_mask_3d = integrated_gradients2.GetMask(
  im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)

baseline2=torch.zeros(1,3,32,32)

black=torch.zeros(3,32,32)
black[0,:,:]=(black[0,:,:]-0.4914)/0.247
black[1,:,:]=(black[1,:,:]-0.4822)/0.243
black[2,:,:]=(black[2,:,:]-0.4465)/0.261
baseline2=black.unsqueeze(0)

mine,b2,c2=guided_ig_torch(im_tensor, baseline2, torch.LongTensor([5]), model, steps=25, fraction=0.1, max_dist=0.02,device='cpu')

guided_ig_mask_3d,b,c = guided_ig2.GetMask(
  im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, max_dist=0.02, fraction=0.1)

guided_ig_mask_3d,b,c 

len(b2)
len(c2)

b2[0]
b[0]


len(b)
len(c)


# Call the visualization methods to convert the 3D tensors to 2D grayscale.

mine.shape
guided_ig_mask_3d.shape
plt.imshow(mine[0].detach().numpy().transpose(1,2,0))
plt.imshow(guided_ig_mask_3d)
torch.movedim(mine[0],0,2).detach().numpy()
mine[0].detach().numpy().transpose(1,2,0)
guided_ig_mask_3d

mine[0].detach().numpy().transpose(1,2,0)

gimg=mine[0].detach().numpy().transpose(1,2,0)
guided_ig_mask_3d

mine[0].detach().numpy().transpose(1,2,0)

vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
guided_ig_mask_grayscale = saliency.VisualizeImageGrayscale(guided_ig_mask_3d)
gimg2=saliency.VisualizeImageGrayscale(mine[0].detach().numpy().transpose(1,2,0))
ROWS = 1
COLS = 3
UPSCALE_FACTOR = 20
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Render the saliency masks.
ShowImage(im_orig, title='Original Image', ax=P.subplot(ROWS, COLS, 1))
ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Integrated Gradients', ax=P.subplot(ROWS, COLS, 2))
ShowGrayscaleImage(guided_ig_mask_grayscale, title='Guided Integrated Gradients', ax=P.subplot(ROWS, COLS, 3))

ShowGrayscaleImage(guided_ig_mask_grayscale, title='Guided Integrated Gradients')
ShowGrayscaleImage(gimg2, title='Guided Integrated Gradients')

def l1_distancenp(x1, x2):
  """Returns L1 distance between two points."""
  return np.abs(x1 - x2).sum()


def translate_x_to_alphanp(x, x_input, x_baseline):
  """Translates a point on straight-line path to its corresponding alpha value.
  Args:
    x: the point on the straight-line path.
    x_input: the end point of the straight-line path.
    x_baseline: the start point of the straight-line path.
  Returns:
    The alpha value in range [0, 1] that shows the relative location of the
    point between x_baseline and x_input.
  """
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.where(x_input - x_baseline != 0,
                    (x - x_baseline) / (x_input - x_baseline), np.nan)


def translate_alpha_to_xnp(alpha, x_input, x_baseline):
  """Translates alpha to the point coordinates within straight-line interval.
   Args:
    alpha: the relative location of the point between x_baseline and x_input.
    x_input: the end point of the straight-line path.
    x_baseline: the start point of the straight-line path.
  Returns:
    The coordinates of the point within [x_baseline, x_input] interval
    that correspond to the given value of alpha.
  """
  assert 0 <= alpha <= 1.0
  return x_baseline + (x_input - x_baseline) * alpha


def guided_ig_implnp(x_input, x_baseline, grad_func, steps=200, fraction=0.25,
    max_dist=0.02):
  """Calculates and returns Guided IG attribution.
  Args:
    x_input: model input that should be explained.
    x_baseline: chosen baseline for the input explanation.
    grad_func: gradient function that accepts a model input and returns
      the corresponding output gradients. In case of many class model, it is
      responsibility of the implementer of the function to return gradients
      for the specific class of interest.
    steps: the number of Riemann sum steps for path integral approximation.
    fraction: the fraction of features [0, 1] that should be selected and
      changed at every approximation step. E.g., value `0.25` means that 25% of
      the input features with smallest gradients are selected and changed at
      every step.
    max_dist: the relative maximum L1 distance [0, 1] that any feature can
      deviate from the straight line path. Value `0` allows no deviation and,
      therefore, corresponds to the Integrated Gradients method that is
      calculated on the straight-line path. Value `1` corresponds to the
      unbounded Guided IG method, where the path can go through any point within
      the baseline-input hyper-rectangular.
  """

  x_input = np.asarray(x_input, dtype=np.float64)
  x_baseline = np.asarray(x_baseline, dtype=np.float64)
  x = x_baseline.copy()
  l1_total = l1_distancenp(x_input, x_baseline)
  attr = np.zeros_like(x_input, dtype=np.float64)
  mid_out=[]
  mid_grad=[]

  # If the input is equal to the baseline then the attribution is zero.
  total_diff = x_input - x_baseline
  if np.abs(total_diff).sum() == 0:
    return attr

  # Iterate through every step.
  for step in range(steps):
    # Calculate gradients and make a copy.
    grad_actual = grad_func(x)
    grad = grad_actual.copy()
    # Calculate current step alpha and the ranges of allowed values for this
    # step.
    alpha = (step + 1.0) / steps
    alpha_min = max(alpha - max_dist, 0.0)
    alpha_max = min(alpha + max_dist, 1.0)
    x_min = translate_alpha_to_xnp(alpha_min, x_input, x_baseline)
    x_max = translate_alpha_to_xnp(alpha_max, x_input, x_baseline)
    # The goal of every step is to reduce L1 distance to the input.
    # `l1_target` is the desired L1 distance after completion of this step.
    l1_target = l1_total * (1 - (step + 1) / steps)

    # Iterate until the desired L1 distance has been reached.
    gamma = np.inf
    while gamma > 1.0:
      x_old = x.copy()
      x_alpha = translate_x_to_alphanp(x, x_input, x_baseline)
      x_alpha[np.isnan(x_alpha)] = alpha_max
      # All features that fell behind the [alpha_min, alpha_max] interval in
      # terms of alpha, should be assigned the x_min values.
      x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

      # Calculate current L1 distance from the input.
      l1_current = l1_distancenp(x, x_input)
      # If the current L1 distance is close enough to the desired one then
      # update the attribution and proceed to the next step.
      if math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
        attr += (x - x_old) * grad_actual
        break

      # Features that reached `x_max` should not be included in the selection.
      # Assign very high gradients to them so they are excluded.
      grad[x == x_max] = np.inf

      # Select features with the lowest absolute gradient.
      threshold = np.quantile(np.abs(grad), fraction, interpolation='lower')
      s = np.logical_and(np.abs(grad) <= threshold, grad != np.inf)

      # Find by how much the L1 distance can be reduced by changing only the
      # selected features.
      l1_s = (np.abs(x - x_max) * s).sum()

      # Calculate ratio `gamma` that show how much the selected features should
      # be changed toward `x_max` to close the gap between current L1 and target
      # L1.
      if l1_s > 0:
        gamma = (l1_current - l1_target) / l1_s
      else:
        gamma = np.inf

      if gamma > 1.0:
        # Gamma higher than 1.0 means that changing selected features is not
        # enough to close the gap. Therefore change them as much as possible to
        # stay in the valid range.
        x[s] = x_max[s]
      else:
        assert gamma > 0, gamma
        x[s] = translate_alpha_to_xnp(gamma, x_max, x)[s]
      # Update attribution to reflect changes in `x`.
      mid_out.append(x)
      mid_grad.append(attr)
      attr += (x - x_old) * grad_actual
  return attr, mid_out, mid_grad




class GuidedIGnp(saliency.GuidedIG):
  """Implements ML framework independent version of Guided IG."""

  def GetMask(self, x_value, call_model_function, call_model_args=None,
      x_baseline=None, x_steps=200, fraction=0.25, max_dist=0.02):

    """Computes and returns the Guided IG attribution.
    Args:
      x_value: an input (ndarray) for which the attribution should be computed.
      call_model_function: A function that interfaces with a model to return
        specific data in a dictionary when given an input and other arguments.
        Expected function signature:
        - call_model_function(x_value_batch,
                              call_model_args=None,
                              expected_keys=None):
          x_value_batch - Input for the model, given as a batch (i.e. dimension
            0 is the batch dimension, dimensions 1 through n represent a single
            input).
          call_model_args - user defined arguments. The value of this argument
            is the value of `call_model_args` argument of the nesting method.
          expected_keys - List of keys that are expected in the output. For this
            method (Guided IG), the expected keys are
            INPUT_OUTPUT_GRADIENTS - Gradients of the output being
              explained (the logit/softmax value) with respect to the input.
              Shape should be the same shape as x_value_batch.
      call_model_args: The arguments that will be passed to the call model
        function, for every call of the model.
      x_baseline: Baseline value used in integration. Defaults to 0.
      x_steps: Number of integrated steps between baseline and x.
      fraction: the fraction of features [0, 1] that should be selected and
        changed at every approximation step. E.g., value `0.25` means that 25%
        of the input features with smallest gradients are selected and changed
        at every step.
      max_dist: the relative maximum L1 distance [0, 1] that any feature can
        deviate from the straight line path. Value `0` allows no deviation and;
        therefore, corresponds to the Integrated Gradients method that is
        calculated on the straight-line path. Value `1` corresponds to the
        unbounded Guided IG method, where the path can go through any point
        within the baseline-input hyper-rectangular.
    """

    x_value = np.asarray(x_value)
    if x_baseline is None:
      x_baseline = np.zeros_like(x_value)
    else:
      x_baseline = np.asarray(x_baseline)

    assert x_baseline.shape == x_value.shape

    return guided_ig_implnp(
        x_input=x_value,
        x_baseline=x_baseline,
        grad_func=self._get_grad_func(call_model_function, call_model_args),
        steps=x_steps,
        fraction=fraction,
        max_dist=max_dist)





def guided_ig_torch(x_input, x_baseline, label ,model, steps=128, fraction=0.25, max_dist=0.02,device='cuda:1'):
    
    x = x_baseline.clone().detach()
    l1_total = l1_distance(x_input, x_baseline)
    attr = torch.zeros_like(x_input,dtype=torch.float32)
    
    total_diff = x_input - x_baseline
    
    mid_out=[]
    mid_grad=[]
    
    
    if torch.abs(total_diff).sum() == 0:
        return attr
    
    for step in range(steps):
        
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32, device=device)
        # Calculate gradients and make a copy.
        grad_actual = guided_compute_gradients(x, model, label)
        grad = grad_actual.clone().detach()
        
        # Calculate current step alpha and the ranges of allowed values for this step.
        
        alpha = (step + 1.0) / steps
        alpha_min = max(alpha - max_dist, 0.0)
        alpha_max = min(alpha + max_dist, 1.0)

        x_min = translate_alpha_to_x(alpha_min, x_input, x_baseline)
        x_max = translate_alpha_to_x(alpha_max, x_input, x_baseline)
        
        # The goal of every step is to reduce L1 distance to the input.
        l1_target = l1_total * (1 - (step+1)/steps)     

        gamma = torch.tensor(float('inf'))
        
        while gamma > 1.0:
            x_old = x.clone().detach()
            x_alpha = translate_x_to_alpha(x, x_input, x_baseline)
            x_alpha = torch.where(torch.isnan(x_alpha),alpha_max,x_alpha)
            
            # All features that fell behind the [alpha_min, alpha_max] interval in
            # terms of alpha, should be assigned the x_min values.
            
            x = torch.where(x_alpha<alpha_min, x_min, x)

            # Calculate current L1 distance from the input.
            l1_current = l1_distance(x, x_input)
            
            # If the current L1 distance is close enough to the desired one then
            # update the attribution and proceed to the next step.
            
            if torch.isclose(l1_target, l1_current, rtol=EPSILON, atol=EPSILON):
                attr += (x - x_old) * grad_actual
                break
            
            # Features that reached `x_max` should not be included in the selection.
            # Assign very high gradients to them so they are excluded.
            grad[x == x_max] = torch.tensor(float('inf'))
            
            threshold = torch.quantile(torch.abs(grad), fraction, interpolation='lower')
            s = torch.logical_and(torch.abs(grad) <= threshold, ~torch.isinf(grad))
            
            # Find by how much the L1 distance can be reduced by changing only the
            # selected features.
            l1_s = (torch.abs(x - x_max) * s).sum()
    
            if l1_s > 0:
                gamma = (l1_current - l1_target) / l1_s
            else:
                gamma = torch.tensor(float('inf'))

            if gamma > 1.0:
                # Gamma higher than 1.0 means that changing selected features is not
                # enough to close the gap. Therefore change them as much as possible to
                # stay in the valid range.
                x[s] = x_max[s]
            else:
                assert gamma > 0, gamma
                x[s] = translate_alpha_to_x(gamma, x_max, x)[s]
            # Update attribution to reflect changes in `x`.    
            mid_out.append(x)
            mid_grad.append(attr)
            attr += (x - x_old) * grad_actual
    return attr, mid_out, mid_grad

























