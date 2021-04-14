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