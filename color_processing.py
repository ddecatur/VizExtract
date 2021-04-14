
def col_dist(c1,c2):
    """A fucntion that calculates the distance between two RGB colors

    """

    # parse colors into R,G,B values
    (r,g,b) = c1
    (R,G,B) = c2
    # return distance formula in r,g,b space
    return ((r-R)**2 + (g-G)**2 + (b-B)**2)**(1/2)

def find_nearest_col(color, dic):
    """A fucntion that takes an RGB color and a dictionary mapping RGB colors
    to strings representing those colors and returns the string corresponding
    to the nearest RBG value in the color map to the given RGB.
    
    """

    minD = 442 # max color distance
    minC = 'empty'
    for elem in dic:
        d = col_dist(elem,color)
        if d < minD:
            minD = d
            minC = dic.get(elem)
    return minC