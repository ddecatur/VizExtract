from create_graph import create_multiData
from numpy.random import choice
from pipeline import process_img
from collections import Counter
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
import json
import matplotlib.pyplot as plt
import numpy as np

NUM_GRAPHS = 100
posSNs = [1,2,3]

def eval_ve(numGraphs, seriesNumber, graphType, folder='chartOcr_images'):
    hits = 0
    aHits = 0
    ctr = 0
    gt_image_to_corr_set = {}
    ve_image_to_corr_set = {}
    for i in range(numGraphs):
        if seriesNumber == "random":
            seriesNum = choice(posSNs)
        else:
            seriesNum = seriesNumber
        ctr += 1
        name,x1s,x2s,tstr,label_to_corr_map,col_to_corr_map = create_multiData(i, seriesNum, "train", graphType, "multi",
            "solid", 'pt', outputDir2=folder, noHard=True)
        iname = name + ".png"
        print(iname)
        gt_c = Counter(label_to_corr_map.values())
        gt_image_to_corr_set[iname] = sorted(gt_c.elements())
        test_string,test_leg_set = process_img(folder + "/" + iname, use_text_not_color=False)
        corrList = []
        for elem in test_leg_set:
            corrList.append(elem.split(": ")[1])
        ve_c = Counter(corrList)
        ve_image_to_corr_set[iname] = sorted(ve_c.elements())
        print(gt_c)
        print(ve_c)
        intersection_c = gt_c & ve_c
        union_c = gt_c | ve_c
        aHits += sum(intersection_c.values()) / sum(union_c.values())
        if (ve_c == gt_c):
            hits += 1

    with open(folder + '_info/ground_truth.txt', 'w') as gt_file:
        gt_file.write(json.dumps(gt_image_to_corr_set))
    with open(folder + '_info/results.txt', 'w') as result_file:
        result_file.write(str(hits/ctr) + "\n" + str(aHits/ctr))
    return hits/ctr


def eval_DeepRule(graphType):
    folder = "chartOcr_images/" + graphType
    with open(folder + '_info/ground_truth.txt') as f:
        gt_data = f.read()
    with open('saves/DeepRule_' + graphType + '.txt') as f:
        dr_data = f.read()
    # with open('saves/simpletest_bar.txt') as f:
    #     dr_data = f.read()
    # with open('saves/simpletest_line.txt') as f:
    #     dr_data = f.read()
    gt = json.loads(gt_data)
    # gt = {"simpleTest2.png": ["neutral"], "graph_2.png": ["positive"], "simple_line_graph.png": ["positive"]}
    # gt = {"simple_test.png": ["neutral"]}
    dr = json.loads(dr_data)
    dr_corr = {}
    
    for graph in dr:
        corrs = []
        if graphType == "line":
            for series in dr[graph]:
                xys = [list(x) for x in zip(*series)]
                xs = xys[0]
                ys = [-y for y in xys[1]]
                func = interp1d(xs, ys)
                xnew = np.arange(min(xs), max(xs), 0.1)
                ynew = func(xnew)
                plt.plot(xs, ys, 'o', xnew, ynew, '-')
                corr = corr, _ = spearmanr(xnew, ynew)
                print(corr)
                if corr >= 0.4:
                    corr = 'positive'
                elif corr <= -0.4:
                    corr = 'negative'
                else:
                    corr = 'neutral'
                corrs.append(corr)
            corrs.sort()
            plt.savefig("inspection_plots/" + graph)
            plt.clf()
        elif graphType == "bar":
            print("graph="+str(graph))
            ys = [-(el[1]+el[3]) for el in dr[graph]]
            print(ys)
            xs = [el[0] for el in dr[graph]] #range(len(ys))
            plt.bar(xs,ys)
            plt.savefig("inspection_plots/" + graph)
            plt.clf()
            corr = corr, _ = spearmanr(xs, ys)
            print(corr)
            if corr >= 0.4:
                corr = 'positive'
            elif corr <= -0.4:
                corr = 'negative'
            else:
                corr = 'neutral'
            corrs.append(corr)
        dr_corr[graph] = corrs
    print(dr_corr)
    hits = 0
    aHits = 0
    ctr = 0
    for graph in gt:
        ctr += 1
        print(graph)
        print(gt[graph])
        print(dr_corr[graph])
        gt_c = Counter(gt[graph])
        dr_c = Counter(dr_corr[graph])
        intersection_c = gt_c & dr_c
        aHits += sum(intersection_c.values()) / max(len(gt[graph]),len(dr_corr[graph]))
        if gt[graph] == dr_corr[graph]:
            hits += 1
        else:
            print("----------Miss----------")
    
    result = {"Score": hits/ctr, "Adjusted Score": aHits/ctr}
    return result


#print(eval_ve(NUM_GRAPHS, "random", "line", folder="chartOcr_images/line"))
print(eval_ve(NUM_GRAPHS, "random", "bar", folder="chartOcr_images/bar"))
#print(eval_DeepRule("bar"))
