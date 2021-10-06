from predict import *
from seg_img import *
from create_graph import find_nearest_col
from PIL import Image
from ocr import *
from object_prediction import *
from numpy.random import random
from color_processing import dist
import cv2
import matplotlib.pyplot as plt
import os
import time

# mapping of rgb values to colors
# used for "find_nearest_col(_, posRGB)"
posRGB = {(255,0,0):'red', (0,128,0):'green', (0,0,255):'blue',
    (255,165,0):'orange', (255,0,255):'purple', (255,255,0):'yellow'}

def permutation(lst):
    """A function that takes a list and returns a list of all possible
        permutations of elements in that list

    """

    if len(lst) == 0: 
        return [] 
  
    if len(lst) == 1: 
        return [lst] 
  
    l = [] # empty list that will store current permutation 
  
    # Iterate the input(lst) and calculate the permutation 
    for i in range(len(lst)): 
       m = lst[i] 
  
       # Extract lst[i] or m from the list.  remLst is 
       # remaining list 
       remLst = lst[:i] + lst[i+1:] 
  
       # Generating all permutations where m is first 
       # element 
       for p in permutation(remLst): 
           l.append([m] + p) 
    return l

def match_series(col_to_seg_map,offset,leg_text_boxes,img_shape,
    algo='current'):
    """A function that matches series text to colors (which are already linked
        to correlation) thus matching series text to correlation

    Parameters:
    -----------
    col_to_seg_map: dict, a mapping from colors to segmentated images (w/ that
        color segmented)
    offset: int, an x coordinate offset corresponding to the number of pixels
        croped out of the legend when removing the key (originally done for OCR)
        This is necessary for when we compute the distance [I think, need to
        recheck this later]
    leg_text_boxes: dict, a mapping of legend text bounding boxes to legend text
    img_shape: dimensions of the image
    algo: str, a tag specifying the algorithm in place (only set to 'old' for
        certain comparison experiemnts)

    Returns:
    --------
    rtn: dict, a mapping of colors to series text
        (in the case of algo='old' it returns the pair (rtn,rtn2)) rtn2: dict, a
        mapping of colors to text but with the old algorithm
    """

    match_series_time = time.time()
    rtn = {}
    rtn2 = {}
    matching = {}
    dist_list = []
    combin = {}
    texts = []
    colors = []
    # iterate through the bounding boxes of the legend text elements
    for elem in leg_text_boxes:
        texts.append(elem)
        currCol = None
        (x,y) = leg_text_boxes[elem]
        pt = (x + offset, y)
        minD = img_shape.shape[0] + img_shape.shape[1]
        colors = []
        # for each color copmute the distance of the given text
        # element to the given key element of that color
        for col in col_to_seg_map:
            coordinates = []
            xcords = []
            ycords = []
            img = col_to_seg_map[col]
            # here using a manual loop through since np.where() not working
            for i in range(0,img_shape.shape[0]): 
                for j in range(0,img_shape.shape[1]):
                    if img[i][j] != 0:
                        xcords.append(j)
                        ycords.append(i)
                        coordinates.append((j,i))
            if xcords != [] and ycords != []:
                colors.append(col)
                xcords.sort()
                ycords.sort()
                xmed = xcords[int(len(xcords)/2)-1]
                ymed = ycords[int(len(ycords)/2)-1]
                distance = dist((xmed,ymed), pt)
                if distance in matching:
                    print("warning: same distance")
                    distance = distance + (random()*0.00000001)
                    # hopefully suffficiently small
                dist_list.append(distance)
                combin[elem+col] = distance
                matching[distance] = (elem, col)

    # sort by the distances (so that we take the smallest distance) 
    dist_list.sort()
    seen_already = []
    
    min_perm_dist = -1 # needs to sufficiently large
    final_perm = []

    # loop through all possible permutations
    for perm in permutation(colors):
        perm_dist = 0
        # for the min of the number of text elements and the number of elements
        # in the given permutation, add up the total distance for all
        # assignments in this permutation to achieve the total "perm_dist"
        for i in range(0,min(len(texts),len(perm))):
            perm_dist = perm_dist + combin[texts[i]+perm[i]]
        # if this permutation had a smaller total distance than any previous
        # permutation set it to be "final_perm". Once we have gone thourgh every
        # possible permutatiion, "final_perm" will hold the permutation that
        # minimizes the total distance
        if min_perm_dist == -1 or (perm_dist < min_perm_dist):
            min_perm_dist = perm_dist
            final_perm = perm
    
    # to ensure that we discard extra information from the permutation step,
    # for every element in the final permutation such that there is a
    # correpsonding element in texts, we add two elements to the "rtn" dict
    # as key and value respectively
    for i in range(0,len(final_perm)):
        if len(texts) >= i+1:
            rtn[final_perm[i]] = texts[i]

    # if we are using the old algorithm, for each element in the dist_list
    # which is sorted in ascending order by distance, we lookup that distance in
    # the matching dictionary to determine the text and color corresponding to
    # that distance. Then if we have not already assigned the given text or
    # color then we append them to the respective "seen" lists and add the color
    # and text to the rtn2 dict as key and value respectively.
    if algo == 'old':
        for dist_elem in dist_list:
            (text, color) = matching[dist_elem]
            if text not in seen_already and color not in seen_already:
                seen_already.append(text)
                seen_already.append(color)
                rtn2[color] = text
        return (rtn,rtn2)
    
    print('match_series_time: ', time.time() - match_series_time)
    return rtn


def run(img, algo='current', use_text_not_color=True):
    """A function to run the other object detection and OCR code on a given
        image array

    Parameters:
    -----------
    img: an image array
    algo: str, a tag specifying the algorithm in place (only set to 'old' for
        certain comparison experiemnts)
    use_text_not_color: bool, a tag to specify whether to use the current or old
        algorithm

    Returns:
    --------
    A 3-tuple of the form (rtn, text_dict, ocr)
    rtn: dict, a mapping of series text to correlations
        (in the case of algo='old' it returns the pair (rtn,rtn2)) rtn2: dict, a
        mapping of series text to correlations, but with the old algorithm
    text_dict: dict, the result of ocr.crop()
    ocr: OCR object
    Note: if "use_text_not_color" is False, then all series mappings are from
        color to correlation not text to correlation
    """

    #create new dir
    path = "./pipeline_batch"
    try:
        os.mkdir(path)
    except OSError:
        print ("Warning: Creation of the directory %s failed, might already exist" % path)
    rtn = {}
    rtn2 = {}
    col_to_cat_map = {}
    col_to_seg_map = {}
    segmentImg_start_time = time.time()
    segImg = segmentImg(img)
    print('segment image total time: ', time.time() - segmentImg_start_time)
    jpgimg = Image.open(img).convert('RGB')
    # convert png to jpg
    newimgp = img.split('.')[:-1]
    newimgp = '.'.join(newimgp) + '.jpg'
    newimgp = "images/" + newimgp.split('/')[-1]
    jpgimg.save(newimgp)
    jpgimg.close()
    object_detection_time = time.time()
    ocr = OCR(img,assign_labels(show_inference(detection_model, newimgp)))
    print('object detection time: ', time.time() - object_detection_time)
    crop_time_v1 = time.time()
    text_dict = ocr.crop()
    print('crop time 1: ', time.time() - crop_time_v1)
    
    # add description here
    for i,(res,col) in enumerate(segImg):
        fname = "pipeline_batch/" + str(i) + ".png"
        plt.imsave(fname, res)
        predict_total_time = time.time()
        cat = predictCategory(fname,
            "models/classification/correlation_classification.h5",
            ['negative', 'neutral', 'positive'])
        print(col)
        print('predict time is: ', time.time() - predict_total_time)
        colstr = "["
        for chanel in col:
            if colstr == "[":
                colstr = colstr + str(chanel)
            else:
                colstr = colstr + ", " + str(chanel)
        
        col = find_nearest_col(col,posRGB)

        #for each segmented thing, find box closest to an existing pixel
        if ocr.leg_box != None:
            newimg = Image.open(fname)
            crp_res = newimg.crop(ocr.leg_box)
            crp_arr = np.asarray(crp_res)
            crp_gray = cv2.cvtColor(crp_arr, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(crp_gray, 0, 255, cv2.THRESH_BINARY +
                cv2.THRESH_OTSU)[1]
            col_to_seg_map[col] = thresh
            img_shape = thresh
        
        # rename duplicate colors
        i = 0
        if col not in col_to_cat_map:
            col_to_cat_map[col] = set()
        col_to_cat_map[col].add(cat)
    if (ocr.leg_box != None) and use_text_not_color:
        if algo == 'old':
            col_to_series_map, col_to_series_map2 = match_series(col_to_seg_map,
                ocr.crop_amount, ocr.leg_text_boxes, img_shape, algo=algo)
            for key in col_to_series_map:
                rtn[col_to_series_map[key]] = col_to_cat_map[key]
            for key in col_to_series_map2:
                rtn2[col_to_series_map2[key]] = col_to_cat_map[key]
        else:
            col_to_series_map = match_series(col_to_seg_map, ocr.crop_amount,
                ocr.leg_text_boxes, img_shape, algo=algo)
            for key in col_to_series_map:
                rtn[col_to_series_map[key]] = col_to_cat_map[key]
    else:
        rtn = col_to_cat_map
    if not rtn: # if rtn is somehow still empty set it to the color mapping
        rtn = col_to_cat_map

    if algo == 'old':
        return (rtn,rtn2,text_dict,ocr)
    else:
        return (rtn,text_dict,ocr)


def process_img(img_path, algo='current', use_text_not_color=True):
    """A function to take in an image path and run it through the entire pipline
    
    Parameters:
    -----------
    img_path: str, the path to the image we want to process
    algo: str, a tag specifying the algorithm in place (only set to 'old' for
        certain comparison experiemnts)
    use_text_not_color: bool, a tag to specify whether to use the current or old
        algorithm

    Returns:
    --------
    A 3-tuple of the form (display_string, corr_set)
    display string: str, a string listing out the extracted axis information
        of the form: "[path_to_img], x axis: [x_axis_text],
                        y axis: [y_axis_text], title: [title_text]"
    corr_set: set, a set containing strings where each string contains a series
        text and and its associated correlation of the form:
        {'[series_text]: negative', '[series_text]: positive'}

    Note: in the case of algo='old', it returns the 4-tuple
        (display_string, corr_set, corr_set2) where corr_set2 is exactly like
        corr_set except it is obtained using the old algorithm specified in
        match_series()

    Note: if "use_text_not_color" is False, then all series mappings
        (e.g. the strings contained in corr_set) are from color to correlation
        not text to correlation
    """

    process_img_start_time = time.time()
    # if the algo is set to 'old', then we also use the old OCR method described
    # in ocr.py
    if algo=='old':
        result,result2,text_dict,OCR = run(img_path, algo=algo)
        text_dict = {'x axis': OCR.xAxisLab, 'y axis': OCR.yAxisLab,
            'title': OCR.title}
        display_string = img_path
        for elem in text_dict:
            if elem != 'legend' and text_dict[elem] is not None:
                display_string = display_string + ", " + elem + ": " + text_dict[elem]
        corr_set = set()
        corr_set2 = set()
        for series in result:
            for elem in result[series]:
                corr_set.add(series + ": " + elem)
        for series in result2:
            for elem in result2[series]:
                corr_set2.add(series + ": " + elem)
        
        return (display_string, corr_set, corr_set2)
    # else we use the normal method
    else:
        run_time_total = time.time()
        result,text_dict,OCR = run(img_path, algo=algo,
            use_text_not_color=use_text_not_color)
        print('run time total: ', time.time() - run_time_total)
        crop_time = time.time()
        text_dict = OCR.crop()
        print('crop time is: ', time.time() - crop_time)
        display_string = img_path
        for elem in text_dict:
            if elem != 'legend' and text_dict[elem] is not None:
                display_string = display_string + ", " + elem + ": " + ' '.join(text_dict[elem])
        corr_set = set()
        for series in result:
            for elem in result[series]:
                corr_set.add(series + ": " + elem)
        
        print('process_img_total_time', time.time() - process_img_start_time)
        return (display_string, corr_set)

# Below is code used for testing individual graphs
# overall_start_time = time.time()
# strrrr, setttt = process_img('images/deeprule_test5.png', use_text_not_color=True)
# print(strrrr)
# print(setttt)
# print('overall_time', time.time() - overall_start_time)