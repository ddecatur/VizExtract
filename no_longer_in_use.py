# A file to house all of the funtions developed during this research
# that did not end up getting used in the final version

# from create_graph.py
def train_series_class(size, dataStyle, directory):
    cwd=os.getcwd()
    if(cwd!=directory):
        print("error: create_data called from wrong directory")
    else:
        # ----------------------------------------
        path = "./series_filtered"
        try:
            os.mkdir(path)
        except OSError:
            print ("Warning: Creation of the directory %s failed, might already exist" % path)

        # create training and validation directories
        sNop = ["1", "2", "3"]
        for n in sNop:
            train_path = "./series_filtered/train/" + n
            try:
                os.makedirs(train_path)
            except OSError:
                print ("Warning: Creation of the directory %s failed, might exist already" % train_path)
            train_path = "./series_filtered/validation/" + n
            try:
                os.makedirs(train_path)
            except OSError:
                print ("Warning: Creation of the directory %s failed, might exist already" % train_path)

        # ----------------------------------------

        for i in range (0, size):
            sNopC = int(choice(sNop))
            create_multiData(i, sNopC, 'train', 'random', 'multi', dataStyle, 's')
            create_multiData(i, sNopC, 'validation', 'random', 'multi', dataStyle, 's')


# from seg_img.py
def find_nearest_col_rgb(color, dic):
    minD = 442 # max color distance
    minC = tuple() # tuple
    for elem in dic:
        d = col_dist(elem,color) #math.sqrt((r-R)^2 + (g-G)^2 + (b-B)^2)
        if d < minD:
            minD = d
            minC = elem#dic.get(elem)
    return minC

def blur(img):
    return cv2.GaussianBlur(img,(5,5),0)

def median_blur(img):
    return cv2.medianBlur(img,5)

def descritize(img):
    data = np.array(img)
    data = data / 255.0
    (h,w,c) = data.shape
    shape = data.shape
    print(data.shape)
    data = data.reshape(h*w, c)
    print(data.shape)
    kmeans = KMeans(n_clusters=13)
    kmeans.fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
    print(shape)
    img_recolored = new_colors.reshape(shape)
    img_recolored = 255 * img_recolored # Now scale by 255
    img_recolored = img_recolored.astype(np.uint8)
    display_img = Image.fromarray(img_recolored)
    display_img.show()

    return img_recolored

def sharpen(img):
    kernal_h = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernal_h)

def eximg(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            (r,g,b) = img[i][j]
            img[i][j] = (0,0,b)
    return img

def sat_max(*cols):
    max_s = 256
    max_col = None
    for (h,s,v) in cols:
        if s < max_s:
            max_s = s
            max_col = (h,s,v)
    return max_col


def orange_to_red(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    new_img = np.empty(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            (h,s,v) = img[i][j]
            if 5<=h and h<=30:
                new_img[i][j] = (0,s,v)
            else:
                new_img[i][j] = (h,s,v)
    new_img = new_img.astype(np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    return new_img

def max_sat_filter(img,n):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    new_img = np.empty(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            max_col = tuple(img[i][j])
            h,s,v = max_col
            if s < 240:
            #if tuple(img[i][j]) != (0,0,255): # to leave white tiles white
                for ii in range(-n,n):
                    for jj in range(-n,n):
                        if i+ii in range(img.shape[0]) and j+jj in range(img.shape[1]):
                            max_col = sat_max(max_col, tuple(img[i+ii,j+jj]))
            new_img[i][j] = max_col
    new_img = new_img.astype(np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    return new_img



def loc_descritize(img, n):
    new = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ii = min(((i//n)*n) + n//2, img.shape[0]-1)
            jj = min(((j//n)*n) + n//2, img.shape[1]-1)
            img[i][j] = img[ii][jj]
    return img

def HOF(img):
    '''
    horz 2d filter -> h_notav
    vert 
    for each pixel, if val == 0,
    '''
    kernal_h = np.array([[0.5,-1,0.5], 
                       [0.5,-1,0.5],
                       [0.5,-1,0.5]])
    kernal_v = np.array([[0.5,0.5,0.5], 
                       [-1,-1,-1],
                       [0.5,0.5,0.5]])
    horz = cv2.filter2D(img, -1, kernal_h) / 255.0
    vert = cv2.filter2D(img, -1, kernal_v) / 255.0
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            img[i][j] = (img[i][j] * abs(horz[i][j])) + (img[i][j+1] * abs(1-horz[i][j]))

    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            img[i][j] = (img[i][j] * abs(vert[i][j])) + (img[i+1][j] * abs(1-vert[i][j]))
    
    return img

# from k_means_clustering
def calculate_WSS(points, kmax): # aka the elbow method -- code adapted from #https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse

def sil_score(points, kmax):
  sil = []

  # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
  for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    labels = kmeans.labels_
    sil.append(silhouette_score(points, labels, metric = 'euclidean'))
  
  idx = 2
  maxsil = 0
  for i in range(len(sil)):
    if sil[i] > maxsil:
      maxsil = sil[i]
      idx = i+2
  return idx

# determine appropriate k (depreciated)
def find_k(x):
  pred = predictCategory(x,'models/series/series_class_model_v3.h5',[1,2,3]) #'series_class_model_mDS_DC_93acc.h5' 'series_class_model_v2.h5'
  return pred+1

def num_diff_cols(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  numCols = 0
  mem = {}
  for i in range(img.shape[0]//5):
    for j in range(img.shape[1]//5):
        h,s,v = img[i*5][j*5]
        key = nc_in_set(h, s, mem)
        if key not in mem:
          mem[key] = 1
        else:
          temp = mem[key]
          del mem[key]
          mem[pair_avg(key,(h,s))] = temp+1
  for elem in mem:
    if mem[elem] > 30:
      numCols += 1
  
  return numCols

def pair_avg(p1,p2):
  x1,y1 = p1
  x2,y2 = p2
  return ((int(x1)+int(x2))//2, (int(y1)+int(y2))//2)

def nc_in_set(h,s,dic):
    """A function to 

    """
    minD = 10 # max color distance
    for key in dic:
        lh = list()
        uh = list()
        low = h-minD
        high = h+minD
        lh.append(max(low,0))
        uh.append(min(high,180))
        lspill = 0-low
        uspill = high-180
        if lspill > 0:
            lh.append(180-lspill)
            uh.append(180)
        if uspill > 0:
            lh.append(0)
            uh.append(uspill)
        for lb,ub in zip(lh,uh):
            if (lb <= key[0] <= ub) and abs(int(s)-int(key[1])) < 10:
                return key
    
    return (h,s)


# these next two are testing functions taken from https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
def centroid_histogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	hist = hist.astype("float")
	hist /= hist.sum()
	return hist

def plot_colors(hist, centroids):
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	for (percent, color) in zip(hist, centroids):
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	return bar

def tests():
    img = './graphs_filtered/testttttt.png'
    graph = cv2.imread(img)
    graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB)
    graph = graph.reshape((graph.shape[0] * graph.shape[1], 3))
    clt = KMeansCluster(graph)
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)
    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()

# from object prediction

#   print(output_dict.keys())
#   print(output_dict['num_detections'])
  #output_dict['detection_boxes']

def predict():
    predictions = []
    for image_path in TEST_IMAGE_PATHS:
        predictions.append(show_inference(detection_model, image_path))
    return predictions

single_img_path = "images/test.jpg" #PATH_TO_TEST_IMAGES_DIR.glob("test.jpg")
# print(single_img_path)
# print(assign_labels(show_inference(detection_model, single_img_path)))

#show_inference(detection_model, single_img_path, 1)


#   # Visualization of the results of a detection.
  
  
#   vis_util.visualize_boxes_and_labels_on_image_array(
#       image_np,
#       output_dict['detection_boxes'],
#       output_dict['detection_classes'],
#       output_dict['detection_scores'],
#       category_index,
#       instance_masks=output_dict.get('detection_masks_reframed', None),
#       use_normalized_coordinates=True,
#       line_thickness=8)
# #   plt.figure()
# #   plt.imshow(Image.fromarray(image_np))
# #   plt.savefig('testod.png')
#   #display(Image.fromarray(image_np))
#   Image.fromarray(image_np).save('predictions/testod'+ str(i) +'.png')

'''
Notes form vis/vis function
'''
#box_to_display_str_map = collections.defaultdict(list)
  #box_to_color_map = collections.defaultdict(str)
  #box_to_instance_masks_map = {}
  #box_to_instance_boundaries_map = {}
  #box_to_keypoints_map = collections.defaultdict(list)
  #box_to_keypoint_scores_map = collections.defaultdict(list)
  #box_to_track_ids_map = {}
 #   if instance_masks is not None:
    #     box_to_instance_masks_map[box] = instance_masks[i]
    #   if instance_boundaries is not None:
    #     box_to_instance_boundaries_map[box] = instance_boundaries[i]
    #   if keypoints is not None:
    #     box_to_keypoints_map[box].extend(keypoints[i])
    #   if keypoint_scores is not None:
    #     box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
    #   if track_ids is not None:
    #     box_to_track_ids_map[box] = track_ids[i]
    #   if scores is None:
    #     box_to_color_map[box] = groundtruth_box_visualization_color
    #   else:
        

        #   if not display_str:
        #     display_score = '{}%'.format(round(100*scores[i]))
        #     #display_str = '{}%'.format(round(100*scores[i]))
        #   else:
        #     #display_str = '{}: {}%'.format(display_str, round(100*scores[i]))
        #     displ
        # if not skip_track_ids and track_ids is not None:
        #   if not display_str:
        #     display_str = 'ID {}'.format(track_ids[i])
        #   else:
        #     display_str = '{}: ID {}'.format(display_str, track_ids[i])
        #box_to_class_and_score_map[box].append((display_class, display_score))
        # if agnostic_mode:
        #   box_to_color_map[box] = 'DarkOrange'
        # elif track_ids is not None:
        #   prime_multipler = _get_multiplier_for_color_randomness()
        #   box_to_color_map[box] = STANDARD_COLORS[
        #       (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
        # else:
        #   box_to_color_map[box] = STANDARD_COLORS[
        #       classes[i] % len(STANDARD_COLORS)]

#   # record coords
#   for box, st in box_to_display_str_map.items():
#       ymin, xmin, ymax, xmax = box
#       print(st[0])
#       #print(st[1])
#       if st[0] not in box_dic:
#           box_dic[st[0]] = set()
#       box_dic[st[0]].add(box)




# ----------------------------------------------
# from ocr.py
# start = datetime.datetime.now()
# segImg = segmentImg('images/test.jpg') # 'test_images/test1.png') #legend_test.png')
# print('segImg elapsed time: ', (datetime.datetime.now()-start).total_seconds())
# start = datetime.datetime.now()
# label_dict = assign_labels(show_inference(detection_model, single_img_path))
# ocr = OCR('images/test.jpg', segImg, label_dict) # 'test_images/test1.png', segImg) #legend_test.png', segImg)
# res = ocr.crop()
# print(res)
# # for i in range(ocr.n_boxes):
# #     if int(ocr.d['conf'][i]) > 60:
# #         (x,y,w,h) = (ocr.d['left'][i], ocr.d['top'][i], ocr.d['width'][i], ocr.d['height'][i])
# #         img = cv2.rectangle(ocr.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# # # cv2.imshow('img', img)
# # # cv2.waitKey(0)
# # ocr.axisLab()
# print('OCR elapsed time: ', (datetime.datetime.now()-start).total_seconds())
# #print(ocr.d['text'])
# print(ocr.xAxisLab)
# print(ocr.yAxisLab)
# ocr.bbDist()
# print(ocr.seriesCorsp)
# for i,elem in enumerate(ocr.di):
#     #print(elem)
#     #print('new dict')
#     for text in elem['text']:
#         #print(text)
#         if (not text.isspace()) and text!='':
#             print(text)
# for i in range(0,ocr.idvdilen):
#     print(ocr.di[i]['text'])
# ocr.mser()

# # ------------ Following preprocessing functions taken from https://nanonets.com/blog/ocr-with-tesseract/ ------------


# # noise removal
# def remove_noise(image):
#     return cv2.medianBlur(image,5)
 


# #dilation
# def dilate(image):
#     kernel = np.ones((5,5),np.uint8)
#     return cv2.dilate(image, kernel, iterations = 1)
    
# #erosion
# def erode(image):
#     kernel = np.ones((5,5),np.uint8)
#     return cv2.erode(image, kernel, iterations = 1)

# #opening - erosion followed by dilation
# def opening(image):
#     kernel = np.ones((5,5),np.uint8)
#     return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# #canny edge detection
# def canny(image):
#     return cv2.Canny(image, 100, 200)

# #skew correction
# def deskew(image):
#     coords = np.column_stack(np.where(image > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return rotated

# #template matching
# def match_template(image, template):
#     return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
# # ------------ End preprocessing functions ------------

def mser(self):
        '''
        METHOD #1
        '''
        _, bw = cv2.threshold(self.img, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy,=cv2.findContours(connected.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        counter=0
        array_of_texts=[]
        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            cropped_image = self.image_obj.crop((x-10, y, x+w+10, y+h ))
            str_store = re.sub(r'([^\s\w]|_)+', '', pytesseract.image_to_string(cropped_image))
            d = pytesseract.image_to_data(cropped_image, config='--psm 12 -c tessedit_char_whitelist=0123456789.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', output_type=Output.DICT)#pytesseract.image_to_string(cropped_image))
            self.di.append(d)
            array_of_texts.append(str_store)
            counter+=1
        self.idvdilen = len(contours)
        #print(array_of_texts)
        #print('len of cont: ' + str(self.idvdilen))
        
        
        self.di.append({'text': ['now on to method #2']})
        '''
        METHOD #2
        '''
        # mser = cv2.MSER_create()
        # vis = self.img.copy()
        # regions, bboxes = mser.detectRegions(self.img)
        # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        # cv2.polylines(vis, hulls, 1, (0, 255, 0))
        # # cv2.imshow('img', vis)
        # # cv2.waitKey(0)
        # mask = np.zeros((self.img.shape[0], self.img.shape[1], 1), dtype=np.uint8)
        # for contour in hulls:
        #     x, y, w, h = cv2.boundingRect(contours[idx])
        #     cropped_image = self.image_obj.crop((x-10, y, x+w+10, y+h ))
        #     cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        #     d = pytesseract.image_to_data(cropped_image, config='--psm 12', output_type=Output.DICT)#pytesseract.image_to_string(cropped_image))
        #     self.di.append(d)
        # text_only = cv2.bitwise_and(self.img, self.img, mask=mask)
        # # cv2.imshow("text only", text_only)
        # # cv2.waitKey(0)


        # vis = self.img.copy()
        # mser = cv2.MSER_create()
        # regions = mser.detectRegions(self.img)
        # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        # cv2.polylines(vis, hulls, 1, (0, 255, 0))
        # cv2.imshow('img', vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # for bbox in bboxes:
        #     (x,y,w,h) = tuple(bbox)
        #     # print(bbox)
        #     display = cv2.rectangle(self.img, (x,y), (x+w, y+h), (0,255,0), 2)
        # print('regions')
        # for reg in regions:
        #     # print(reg)
        #     print(pytesseract.image_to_string(reg))

        # cv2.imshow('display', display)
        # cv2.waitKey(0)


# noise removal
def remove_noise(self):
    self.img = cv2.GaussianBlur(self.img,(11,11),0)

#thresholding
def thresholding(self):
    self.img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def bbDist(self):
        for i in range(0, self.n_boxes):
            if int(self.d['conf'][i]) > 60 and (not self.d['text'][i].isspace()) and self.d['text'][i]!='':
                print(self.d['text'][i])
                minD = self.dimensions[0] + self.dimensions[1]
                pt = (self.d['left'][i], self.d['top'][i])
                coordinates = {}
                currCol = [-1]
                for col in self.seg:
                    indices = np.where(self.seg[col] != [0])
                    coordinates[col] = zip(indices[0], indices[1])
                    for cord in coordinates[col]:
                        dist = self.dist(cord, pt)
                        if dist <= minD:
                            minD = dist
                            currCord = cord
                            currCol = col
                if currCol!=[-1]:
                    self.seriesCorsp[currCol] = self.d['text'][i]
                else:
                    print('never found min distance')

# from pipeline.py
# if col == 'bdfjhdjfhd':
            #     for col in img:
            #         for row in col:
            #             specialph.append(row)
            #     #indices = np.where(img != [0], [255], [0])
            #     #mask_img = Image.fromarray(indices, 'RGB')
            #     mask = np.array(indices, dtype=np.uint8)

            #     cv2.imshow('mask', mask)
            #     cv2.waitKey()
            #     #mask_img.show()
            # else:
            #     #indices = np.where(img != [0])
            #     aaa = 2
# if distance < minD:
                #     minD = distance
                #     currCol = col
                # if col == 'b':
                #     print(distance)
                #     print((xmed,ymed))
                #     print(pt)
                #     return 1
            # print("after done color")
            # print(col)
            # print(minD)
            #print(indices)
            #break
            # print(len(indices[0]))
            # print(len(indices[1]))
            # coordinates = zip(indices[0], indices[1])
            #len(coordinates)
            # for cord in coordinates:
            #     distance = dist(cord, pt)
            #     if distance < minD:
            #         minD = distance
            #         currCol = col
            #         if col == 'b':
            #             print(distance)
            #             print(cord)
            #             print(pt)
            #             return 1
            # print("after done color")
            # print(col)
            # print(minD)
    
        # if currCol!=None:
        #     rtn[currCol] = elem
        # else:
        #     print('never found min distance')
        #print(currCol)
        #print(rtn[currCol])
    #print(dist_list)

# ---------------------- OLD ALGO ---------------------
    # for dist_elem in dist_list:
    #     (text, color) = matching[dist_elem]
    #     if text not in seen_already and color not in seen_already:
    #         seen_already.append(text)
    #         seen_already.append(color)
    #         rtn[color] = text

#if ocr.match_leg_img:
        #print('yes')

        #ocr.match_leg_img.save("legend_cropped.jpg")
        #segLeg = segmentImg("legend_cropped.jpg",fixed_k=len(segImg))
    # print(text_dict)
    # print('segimg len:')
    # print(len(segImg))
    #color_list = []

# if col == 'g':
        #     for col in img_shape:
        #         for row in col:
        #             image_holder.append(row)

        #     # write csv files
        #     with open('./image_holder.csv', 'w') as f:
        #         writer = csv.writer(f, delimiter=',')
        #         writer.writerows(image_holder)
        #     f.close()

        # if col in rtn:
        #     col=col+str(i)
        #     colorMap[col]=col
        #     print('pipeline error: overwrite color')
        # rtn[variable] = cat
        #rtn.append(cat)

    
    
    # print('col_to_series_map')
    # print(col_to_series_map)
    

    # color_list = []
    # if segLeg!=None:
    #     print("yes2")
    #     for res,col in segLeg:
    #         fname = "pipeline_batch/test_leg.png"
    #         plt.imsave(fname, res)
    #         color_list.append(avg_height(res))
    # color_list.sort(reverse=True)
    # print(color_list)
    #print(col_to_cat_map)

# for res,col in segImg:
            # self.seg[tuple(col)] = res
            # color_list.append(match_series(np.asarray(crp_res), ocr.crop_amount, ocr.leg_text_boxes)) # added the crop amount here to be able to recover the coordinates of the text boxes

