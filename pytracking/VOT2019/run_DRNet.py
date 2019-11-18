
import vot
import sys
import time
import cv2 as cv
import os
import numpy
import collections
from pytracking.tracker.drnet.drnet import *
from pytracking.parameter.drnet.default_vot import *
import random
import numpy as np
# del os.environ['MKL_NUM_THREADS']
seed=888
random.seed(seed)
np.random.seed(seed*2)
torch.manual_seed(seed*3)
torch.cuda.manual_seed(seed*4)
torch.backends.cudnn.deterministic = True
def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h


print(2333333)
handle = vot.VOT("polygon")
selection = handle.region()
selection=selection.points
gt_bbox =[]
for i in selection:
    gt_bbox.append(i.x)
    gt_bbox.append(i.y)
if len(gt_bbox) == 4:
    gt_bbox = [gt_bbox[0], gt_bbox[1],
                gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]] 
cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
gt_bbox = [cx-(w-1)/2, cy-(h-1)/2, w, h]
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)
image = cv.cvtColor(cv.imread(imagefile), cv.COLOR_BGR2RGB)
params = parameters()
tracker = DRNet(params)
tracker.initialize(image, list(gt_bbox))
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv.cvtColor(cv.imread(imagefile), cv.COLOR_BGR2RGB)
    region = tracker.track(image)
    region = vot.Rectangle(region[0],region[1],region[2],region[3])
    handle.report(region, 0.9)
