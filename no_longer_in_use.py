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