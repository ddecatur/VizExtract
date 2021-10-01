import matplotlib.pyplot as plt
import numpy as np
from random import choice as randomchoice
import string
from numpy.random import randint
from numpy.random import random
from numpy.random import randn
from numpy.random import choice
from numpy.random import seed
from numpy import cov
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from create_graph import *
import csv
import glob
from pipeline import *
from seg_img import sat_thresh_filter
import yaml
from time import time
from statistics import median, mean

IMG_HEIGHT = 480
IMG_WIDTH = 640

memoization = {}

def editfast(s,t):
    
    # change both to lower case to account for small case errors which are not important to distinguish between
    s = s.lower()
    t = t.lower()

    if (s,t) in memoization:
        return memoization[(s,t)]
    
    if s == "":
        return len(t)
    
    if t == "":
        return len(s)
    
    rtn = min([1 + editfast(s[1:], t), 1 + editfast(s, t[1:]), (s[ 0 ] != t[ 0 ]) + editfast(s[ 1 :], t[ 1 :])])
    
    memoization[(s,t)] = rtn
    
    return rtn


def writeOutput(numGraphs, pstyle="default", txtcolres=True):
    ground_truth = []
    ground_truth.append(("filename","x axis label","y axis label","title string","legend and correlation"))
    
    posSNs = [1,2,3]
    axis_hits = 0
    series_hits = 0
    hits = 0
    ttimes = []
    ctr = 0
    for i in range(0,numGraphs):
        ctr+=1
        leg_display_str = ""
        name,x1s,x2s,tstr,label_to_corr_map,col_to_corr_map = create_multiData(i, choice(posSNs), "train", "random", "multi", "solid", 'pt', pstyle=pstyle, noHard=True)
        iname = name + ".png"
        display_string = "images/" + iname + ", x axis: " + x1s + ", y axis: " + x2s + ", title: " + tstr
        leg_set = set()
        if not txtcolres:
            label_to_corr_map = col_to_corr_map
        for key in label_to_corr_map:
            leg_set.add(key + ": " + label_to_corr_map[key])
            leg_display_str = leg_display_str + ", " + key + ": " + label_to_corr_map[key]
        ground_truth.append((iname, x1s, x2s, tstr, leg_display_str))
        #display_string = display_string + leg_display_str
        t0 = time()
        test_string,test_leg_set = process_img("images/" + iname, use_text_not_color=txtcolres)
        ttime = time()-t0
        ttimes.append(ttime)
        print("time: ", ttime)
        print(display_string)
        print(leg_set)
        print(test_string)
        print(test_leg_set)
        edds = editfast(display_string, test_string)
        if edds < 6:
            axis_hits = axis_hits + 1
        else:
            print('axis fail --------------------------------')
        
        oldsh = series_hits
        for elem1 in leg_set:
            for elem2 in test_leg_set:
                memoization = {}
                edsm = editfast(elem1,elem2)
                #print('edsm =' + str(edsm))
                if edsm < 3:
                    series_hits = series_hits + (1/len(leg_set))
                    break
        if oldsh == series_hits:
            print('series fail --------------------------------')
        if not bool(test_leg_set):
            print('empty set for legend')
        print("axis score: " + str(axis_hits/ctr))
        print("series score: " + str(series_hits/ctr))
        print("total score: " + str(((axis_hits + series_hits)/2/ctr)*100) + "%")

    # time info
    print("Median time: ", median(ttimes))
    print("Mean time: ", mean(ttimes))

    hits = (axis_hits + series_hits)/2
    print("axis score: " + str(axis_hits/numGraphs))
    print("series score: " + str(series_hits/numGraphs))
    score = (hits/numGraphs)*100
    print("total score: " + str(score) + "%")

    # write csv files
    with open('./ground_truth.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(ground_truth)
    f.close()


def test_series_classification(numGraphs):
    posSNs = [1,2,3]
    ks = []
    sNs = []
    for i in range(numGraphs):
        sN = choice(posSNs)
        print(sN)
        sNs.append(sN)
        name,x1s,x2s,tstr,label_to_corr_map = create_multiData(i, sN, "train", "random", "multi", "solid", 'pt')
        iname = name + ".png"
        print(iname)
        img = cv2.imread("images/" + iname)
        img = sat_thresh_filter(img,40)
        ks.append(elbowM(img)-1)
        im = Image.fromarray(img)
        im.save('images/thresh_ref'+str(i)+'.png')
    score = 0
    print(ks)
    print(sNs)
    for i in range(len(ks)):
        if ks[i] == sNs[i]:
            score +=1
    print(score/len(ks))


def test_outside_data():
    mScore = 0
    rScore = 0
    mAScore = 0
    rAScore = 0
    with open('outside_data_labels.yaml') as f:
        gtLabels = yaml.load(f, Loader=yaml.FullLoader)
    M = gtLabels.get('M')
    R = gtLabels.get('R')
    fileList = glob.glob("exp_testing/M/M*.jpg")
    fileList.sort()
    ctr = 0
    for imagePath in fileList:
        print(imagePath)
        ctr += 1
        print(ctr)
        s = imagePath.find('/M/M') + 3
        e = imagePath.find('.', s)
        name = imagePath[s:e]
        gtCorr = set(M[name].keys())
        testAxis,testCorr = process_img(imagePath, use_text_not_color=False)
        print(gtCorr)
        print(testCorr)
        if gtCorr == testCorr:
            mScore += 1
        else:
            with open('Mtestinginfo.csv', 'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow((imagePath, gtCorr, testCorr))
            f.close()
        for elem1 in gtCorr:
            for elem2 in testCorr:
                if elem1 == elem2:
                    mAScore += (1/max(len(gtCorr),len(testCorr)))
        print('M Score: ' + str(mScore/ctr))
        print('M Score Adjusted: ' + str(mAScore/ctr))
    mScore = mScore/ctr
    mAScore = mAScore/ctr
    fileList = glob.glob("exp_testing/R/R*.jpg")
    fileList.sort()
    ctr = 0
    for imagePath in fileList:
        print(imagePath)
        ctr += 1
        print(ctr)
        s = imagePath.find('/R/R') + 3
        e = imagePath.find('.', s)
        name = imagePath[s:e]
        gtCorr = set(R[name].keys())
        testAxis,testCorr = process_img(imagePath, use_text_not_color=False)
        print(gtCorr)
        print(testCorr)
        if gtCorr == testCorr:
            rScore += 1
        else:
            with open('Rtestinginfo.csv', 'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow((imagePath, gtCorr, testCorr))
            f.close()
        for elem1 in gtCorr:
            for elem2 in testCorr:
                if elem1 == elem2:
                    rAScore += (1/max(len(gtCorr),len(testCorr))) # this way it punishes extra info
        print('R Score: ' + str(rScore/ctr))
        print('R Score Adjusted: ' + str(rAScore/ctr))
    rScore = rScore/ctr
    rAScore = rAScore/ctr
    print('M Score: ' + str(mScore))
    print('M Score Adjusted: ' + str(mAScore))
    print('R Score: ' + str(rScore))
    print('R Score Adjusted: ' + str(rAScore))
    print('Total Score: ' + str((mScore+rScore)/2))
    print('Total Adjusted: ' + str((mAScore+rAScore)/2))
    print('Real total score adjusted: ' + str(((mAScore*50)+(rAScore*33)) / 83))


def test_google_img_data():
    eScore = 0
    hScore = 0
    eAScore = 0
    hAScore = 0
    with open('google_img_labels.yaml') as f:
        gtLabels = yaml.load(f, Loader=yaml.FullLoader)
    E = gtLabels.get('easy')
    H = gtLabels.get('hard')
    fileList = glob.glob("google_imgs/easy/*.jpg")
    fileList.sort()
    ctr = 0
    for imagePath in fileList:
        print(imagePath)
        ctr += 1
        print(ctr)
        s = imagePath.find('/easy/') + 6
        e = imagePath.find('.', s)
        name = 'E' + imagePath[s:e]
        gtCorr = set(E[name].keys())
        testAxis,testCorr = process_img(imagePath, use_text_not_color=False)
        print(gtCorr)
        print(testCorr)
        if gtCorr == testCorr:
            eScore += 1
        else:
            with open('easyMissed.csv', 'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow((imagePath, gtCorr, testCorr))
            f.close()
        for elem1 in gtCorr:
            for elem2 in testCorr:
                if elem1 == elem2:
                    eAScore += (1/max(len(gtCorr),len(testCorr)))
        print('E Score: ' + str(eScore/ctr))
        print('E Score Adjusted: ' + str(eAScore/ctr))
    #eScore = eScore/ctr
    #eAScore = eAScore/ctr
    fileList = glob.glob("google_imgs/hard/*.jpg")
    fileList.sort()
    ctrH = 0
    for imagePath in fileList:
        print(imagePath)
        ctrH += 1
        print(ctrH)
        s = imagePath.find('/hard/') + 6
        e = imagePath.find('.', s)
        name = 'H' + imagePath[s:e]
        gtCorr = set(H[name].keys())
        testAxis,testCorr = process_img(imagePath, use_text_not_color=False)
        print(gtCorr)
        print(testCorr)
        if gtCorr == testCorr:
            hScore += 1
        else:
            with open('hardMissed.csv', 'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow((imagePath, gtCorr, testCorr))
            f.close()
        for elem1 in gtCorr:
            for elem2 in testCorr:
                if elem1 == elem2:
                    hAScore += (1/max(len(gtCorr),len(testCorr))) # this way it punishes extra info
        print('H Score: ' + str(hScore/ctrH))
        print('H Score Adjusted: ' + str(hAScore/ctrH))
    #hScore = hScore/ctrH
    #hAScore = hAScore/ctrH
    print('E Score: ' + str(eScore/ctr))
    print('E Score Adjusted: ' + str(eAScore/ctr))
    print('H Score: ' + str(hScore/ctrH))
    print('H Score Adjusted: ' + str(hAScore/ctrH))
    print('Total Score: ' + str((eScore+hScore)/2))
    print('Total Adjusted: ' + str((eAScore+hAScore)/2))
    print('Real total score adjusted: ' + str((eAScore+hAScore) / (ctr+ctrH)))

#test_google_img_data()
writeOutput(100, txtcolres=False)