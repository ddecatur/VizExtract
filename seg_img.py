import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from k_means_clustering import *
from color_processing import *
import math


def sat_thresh_filter(img,thresh):
    """A function to threshold the pixels in an image based
        on the saturation of the given pixel

    For each pixel (in HSV format), if the saturation value is
    above the threshold, set both the saturation and value to
    255 to increase contrast and limit differences within a specific
    hue. If the saturation is below the threshold set the saturation
    to 0 and the value to 255. This changes essentially changes the
    pixel to be white.

    Parameters:
    -----------
    img: an openCV image array
    threshold: int, a threshold for saturation values

    Returns:
    --------
    new_img: the new openCV image array resutling from the thresholding
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    new_img = np.empty(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            (h,s,v) = img[i][j]
            if s > thresh:
                new_img[i][j] = (h,255,255)
            else:
                new_img[i][j] = (h,0,255)
    new_img = new_img.astype(np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    return new_img

def isGrayish(color):
    """A function to determin if the given color is "grayish",
        essentially are the r,g,b values of the given color
        close enought to each other that the color could be
        considered similar to grayscale (where all three r,g,b
        values are identical)

    """

    (r,g,b) = color
    ac = (r+g+b)/3
    cd = col_dist((ac,ac,ac), color)
    if cd < 20:
        return True
    else:
        return False

# type: img path --> list of color ranges
def idColor(image, preprocess=True):
    """A function to determine and identify the colors corresponing to series
        in the image of the graph

    Parameters:
    -----------
    image: str, the filepath to the image
    preprocess: bool, whether or not to perform the preprocessing step
        the false setting of this parameter is primarily used for experiments

    Returns:
    --------
    colList: list, a list 3-tuples containing (r,g,b) values for each color
        corresponding to a series in the image of the graph
    """

    ogImg = image
    # convert image
    image = cv2.imread(image)
    if preprocess:
        image = sat_thresh_filter(image,40)

    clt = KMeansCluster(image,ogImg) # maybe come back and simplify this
    hist = clusterCounts(clt)
    # determine which colors to segment out
    colList = list()
    maxCol = max(hist) # identify the background color as the most common color so it can be removed
    for (percent, color) in zip(hist, clt.cluster_centers_):
        if (percent != maxCol and isGrayish(color)==False): # accpet colors that are not the background color or "grayish"
            colList.append(color.astype("uint8").tolist())
    return colList

def rgb_hsv(r,g,b):
    """A function to convert an r,g,b value into the openCV representation of h,s,v

    code adapted from: https://www.geeksforgeeks.org/program-change-rgb-color-model-hsv-color-model/
    to work for the openCV representation of h,s,v
    """

    # R, G, B values are divided by 255 
    # to change the range from 0..255 to 0..1: 
    r, g, b = r / 255.0, g / 255.0, b / 255.0
  
    # h, s, v = hue, saturation, value 
    cmax = max(r, g, b)    # maximum of r, g, b 
    cmin = min(r, g, b)    # minimum of r, g, b 
    diff = cmax-cmin       # diff of cmax and cmin. 
  
    # if cmax and cmax are equal then h = 0 
    if cmax == cmin:  
        h = 0
      
    # if cmax equal r then compute h 
    elif cmax == r:  
        h = (60 * ((g - b) / diff) + 360) % 360
        h = h/2
  
    # if cmax equal g then compute h 
    elif cmax == g: 
        h = (60 * ((b - r) / diff) + 120) % 360
        h = h/2
  
    # if cmax equal b then compute h 
    elif cmax == b: 
        h = (60 * ((r - g) / diff) + 240) % 360
        h =h/2
  
    # if cmax equal zero 
    if cmax == 0: 
        s = 0
    else: 
        s = (diff / cmax) * 100
    s = (s/100)*255
  
    # compute v 
    v = cmax * 100
    v = (v/100)*255
    return (h,s,v)

def find_ranges(h,s,v):
    """A function to find the range of "close" h,s,v values to
        extract given some starting color

    We extract all pixels within this range in order to pick up
    all pixels corresponding to the series not just those that are
    the exact same color (have the exact same h,s,v value)

    Returns:
    --------
    rtn: set, a set of all ranges that correspond to the given color
    """
    
    rtn = set()
    lh = list()
    uh = list()
    low = h-10
    high = h+10
    lh.append(max(low,0))
    uh.append(min(high,180))
    lspill = 0-low
    uspill = high-180
    # here we have to deal with "overflow" of h values since it ranges from 0-180 degrees
    # but wraps around. For example the range around an h value of 5 needs to include
    # 0-15 and 170-180.
    if lspill > 0:
        lh.append(180-lspill)
        uh.append(180)
    if uspill > 0:
        lh.append(0)
        uh.append(uspill)
    sl = max(s-25,0)
    vl = max(v-40,76.5) # should be -40 # 
    su = min(s+25,255) # should be +10
    vu = min(v+40,255)
    for i,bound in enumerate(lh):
        rtn.add(((bound, sl, vl), (uh[i], su, vu)))
    return rtn
    
    

# given a list of rgb colors return a list of hsv ranges
def hsvRange (rgbList):
    """A function that given a list of rgb colors returns a
        list of sets of hsv ranges for each color in the original list

    """

    rangeList = list()
    for (r,g,b) in rgbList:
        (h,s,v) = rgb_hsv(r,g,b)
        rangeList.append(find_ranges(h,s,v))

    return rangeList


def segmentImg(img, fixed_k=None, preprocess=True):
    '''A function that, givn an image path, segments out colors from that image
    '''

    # read in image
    graph = cv2.imread(img)
    if preprocess:
        graph = sat_thresh_filter(graph,40)
    
    # id colors for each series and their corresponding hsv ranges
    colList = idColor(img, preprocess=preprocess)
    colRangeList = hsvRange(colList)

    # convert to hsv image type
    hsv_graph = cv2.cvtColor(graph, cv2.COLOR_RGB2HSV)
    
    # create new image arrays for each color where all pixels in the
    # corresponding range are segmented out (and background is all black pixels)
    results = list()
    for i,rngSet in enumerate(colRangeList):
        masks = list()
        for (ll,ul) in rngSet:
            masks.append(cv2.inRange(hsv_graph, ll, ul))
        maskFinal = masks[0]
        for j in range(1,len(masks)):
            maskFinal = maskFinal + masks[j]
        results.append((cv2.bitwise_and(graph, graph, mask=maskFinal), colList[i]))

    return results