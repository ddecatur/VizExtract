import cv2
import pytesseract
import re
from pytesseract import Output
from tensorflow.core.protobuf.struct_pb2 import NoneValue
from PIL import Image
from object_prediction import *
import re

class OCR():
    """This class creates an OCR object for given image and its bounding boxes

    Parameters:
    -----------
    image: an image filepath
    box_dict: dict, a dictionary containing the result of "assign_labels()"
        in "object_prediction.py"

    Returns:
    --------
    res: string, a randomly generated string of ASCII letters of length "length"
    """

    def __init__(self, image, box_dict, k=None):
        """Initialize the OCR class

        Parameters:
        -----------
        image: an image filepath
        box_dict: dict, a dictionary containing the result of "assign_labels()"
            in "object_prediction.py"
        """
        
        self.image_obj = Image.open(image)
        self.box_dict = box_dict
        self.match_leg_img = None
        self.leg_box = None
        self.leg_text_boxes = {}
        self.crop_amount = 35
        self.k = k
        self.xAxisLab = None
        self.yAxisLab = None
        self.title = None
        self.legend = []
        self.d = None
        # these following lines are only used for algorithm experimentation
        # purposes and thus can be commented out for now to save time
        # ------------------------------------------------------------------
        # self.img = cv2.imread(image)
        # self.img = cv2.resize(self.img, (0,0), fx=3, fy=3)
        # self.get_grayscale()
        # self.dimensions = self.img.shape
        # self.d = pytesseract.image_to_data(self.img,
        #     config='--psm 3 -c tessedit_char_whitelist=0123456789-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
        #     output_type=Output.DICT)
        # self.get_boxes()
        # ------------------------------------------------------------------
    
    def isREGEX(self, string):
        """A function to determine if a given string does not
            contain any of the following characters

        """
        regex = re.compile('[.@=»_!#$%^&*()<>?/\|}{~:]') # adding hyphons to prevent dashed legend lines from confusing the ocr
        return (regex.search(string) == None)

    def remove_key(self, box):
        """A function to remove the key from the bounding box containing
            the legend so that only the text describing the series names
            remians

        """

        (xmin, ymin, xmax, ymax) = box
        return (xmin+self.crop_amount, ymin, xmax, ymax)

    def crop(self):
        """A function to crop the overall image to specific bounding boxes identified
            as objects and apply OCR to them to obtain the text contained within each one
        """

        text_dict = {}
        self.match_leg_img = None
        for box in self.box_dict:
            crp_img = self.image_obj
            if box == None:
               self.box_dict[box] = 'extra text'
            crp_img = crp_img.crop(box)
            if self.box_dict[box] == 'y axis':
                crp_img = crp_img.rotate(270, expand=True) # might need to be more general in the future
            if self.box_dict[box] == 'legend':
                leg_crop = self.image_obj
                leg_crop = leg_crop.crop(self.remove_key(box))
                self.match_leg_img = crp_img
                self.leg_box = box
                crp_img = leg_crop
            crpD = pytesseract.image_to_data(crp_img,
                config='--psm 3 -c tessedit_char_whitelist=0123456789-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
                output_type=Output.DICT)
            n_boxes = len(crpD['text'])
            strlist = []
            for i in range(0,n_boxes):
                elem = crpD['text'][i]
                if (not elem.isspace()) and elem!='' and elem!=',' and elem!='-' and elem!='—' and elem!='_': 
                    strlist.append(elem)
                    if self.box_dict[box] == 'legend':
                        self.leg_text_boxes[elem] = (crpD['left'][i] + (crpD['width'][i])/2, crpD['top'][i] + (crpD['height'][i])/2) # 

            text_dict[self.box_dict[box]] = strlist
        return text_dict


    def get_grayscale(self):
        """A function to convert an image to grayscale
        """

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)



    # algorithmic label assignment for experimentation purposes only
    # --------------------------------------------------------------
    def dist(self,p1,p2):
        """A function to find the distance between two points
        """

        return (((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2))**(1/2)

    
    def get_boxes(self):
        """A function to algorithmically assign text to graph elements
            (without using object detection)

        """

        maxY = self.dimensions[0]
        xMaxDP, yMaxDP, yMaxDP2 = 0, 0, 0
        xMidPos = self.dimensions[1]//2
        yMidPos = self.dimensions[0]//2
        yMaxDPIdx = None
        yMaxDPIdx2 = None
        xMaxDPIdx = None
        maxYIdx = None
        minDPIdx = None
        minDP = max(self.dimensions[0],self.dimensions[1])
        n_boxes = len(self.d['text'])
        for i in range(n_boxes):
            elem = self.d['text'][i]
            if (not elem.isspace()) and elem!='' and elem!=',' and elem!='-' and elem!='—' and elem!='_': 
                (x,y) = (self.d['left'][i] + (self.d['width'][i])/2, self.d['top'][i] + (self.d['height'][i])/2)
                if abs(x-xMidPos) >= xMaxDP:
                    xMaxDP = abs(x-xMidPos)
                    xMaxDPIdx = i
                if y < maxY: # because reverse coordinates
                    maxY = y
                    maxYIdx = i
                if abs(y-yMidPos) >= yMaxDP:
                    yMaxDP2 = yMaxDP
                    yMaxDPIdx2 = yMaxDPIdx
                    yMaxDP = abs(y-yMidPos)
                    yMaxDPIdx = i
                elif abs(y-yMidPos) >= yMaxDP2:
                    yMaxDP2 = abs(y-yMidPos)
                    yMaxDPIdx2 = i
                dst = self.dist((x,y), (xMidPos,yMidPos))
                if dst < minDP:
                    minDP = dst
                    minDPIdx = i
        if xMaxDPIdx is not None:
            self.yAxisLab = self.d['text'][xMaxDPIdx]
        if yMaxDPIdx is not None:
            if yMaxDPIdx != maxYIdx:
                self.xAxisLab = self.d['text'][yMaxDPIdx]
            elif yMaxDPIdx2 is not None:
                self.xAxisLab = self.d['text'][yMaxDPIdx2]
        if maxYIdx is not None:
            self.title = self.d['text'][maxYIdx]


    def axisLab(self):
        """A function to assign text to the respective axies label
            (x axis label or y axis label)

        """

        xMin, yMax = self.dimensions[1], 0
        for i in range(self.n_boxes):
            if int(self.d['conf'][i]) > 60 and (not self.d['text'][i].isspace()):
                (x,y) = (self.d['left'][i], self.d['top'][i])
                if x <= xMin:
                    xMin = x
                    xMinIdx = i
                if y >= yMax:
                    yMax = y
                    yMaxIdx = i
        if (not xMinIdx or not yMaxIdx):
            print('error: ocr -- min or max not set')
        if xMinIdx == yMaxIdx:
            print('x and y label are same text')
            self.xAxisLab = self.d['text'][xMinIdx]
            self.yAxisLab = self.d['text'][yMaxIdx]
        else:
            self.xAxisLab = self.d['text'][xMinIdx]
            self.yAxisLab = self.d['text'][yMaxIdx]
