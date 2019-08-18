import numpy as np
from numpy import linalg
import cv2
from math import sqrt
from math import atan
from math import pow
import glob
import Image
from PIL import Image

q = 1
# for img in glob.glob("Farewell2k17/*.jpg"):
a = cv2.imread("Farewell2k17/2.jpg")
b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
c = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
d = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
k=0.1
if(q==7 or q==8):
    k=0.2
# if(q==1):
#     window=65
# if(q==9 or q==11):
#     k=0.12
# if(q==8):
#     window=25
# if(q!=1  and q!=8):
window=35
cv2.imwrite('hello3.jpg',b)
# thresh=128
# img = cv2.imread('2.jpg',0)
# img = cv2.medianBlur(img,5)
# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
#
#
# cv2.imwrite('binarized2.jpg', th3)
ret2= cv2.threshold(b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
h,w=b.shape
th=0
r=128
value=[]
threshold=[]
m=0
var=0
std=0
b,c=cv2.integral2(b)
cv2.imwrite("integral.jpg",b)
h,w=c.shape
print(q)

# for i in range(0, w - 10, 1):
#     for j in range(0, h - 10, 1):
#         b[j,i]=b[j,i]+b[j-1,i]+b[j,i-1]-b[j-1,i-1]
# for i in range(0, w - 10, 1):
#     for j in range(0, h - 10, 1):
#         c[j,i]=((b[j,i]*b[j,i])+(b[j-1,i]*b[j-1,i])+(b[j,i-1]*b[j,i-1])-(b[j-1,i-1]*b[j-1,i-1]))


for i in range(1, w , 1):
    for j in range(1, h , 1):
        if(j>(h-(window/2)) and i>(w-(window/2))):
            m = (b[j, i] + b[j - (window / 2), i - (window / 2)] - b[
                j , i - (window / 2)] - b[j - (window / 2), i]) / (window * window)
            s = (c[j, i] + c[j - (window / 2), i - (window / 2)] - c[
                j, i - (window / 2)] - c[j - (window / 2), i]) / (window * window)
        elif (i > (w - (window / 2)) and j < (h - (window / 2))):
            m = (b[j + (window / 2), i] + b[j - (window / 2), i - (window / 2)] - b[
                j + (window / 2), i - (window / 2)] - b[j - (window / 2), i]) / (window * window)
            s = (c[j + (window / 2), i] + c[j - (window / 2), i - (window / 2)] - c[
                j + (window / 2), i - (window / 2)] - c[j - (window / 2), i]) / (window * window)
        elif(j>(h-(window/2)) and i<(w-(window/2))):
            m = ( b[j,i+(window/2)]+b[j-(window/2),i-(window/2)]-b[j,i-(window/2)]-b[j-(window/2),i+(window/2)])/(window*window)
            s = (  c[j,i+(window/2)]+c[j-(window/2),i-(window/2)]-c[j,i-(window/2)]-c[j-(window/2),i+(window/2)])/(window*window)
        elif(j<(h-(window/2) ) and i<(w-(window/2))):
            m=(b[j+(window/2),i+(window/2)]+b[j-(window/2),i-(window/2)]-b[j+(window/2),i-(window/2)]-b[j-(window/2),i+(window/2)])/(window*window)
            s=(c[j+(window/2),i+(window/2)]+c[j-(window/2),i-(window/2)]-c[j+(window/2),i-(window/2)]-c[j-(window/2),i+(window/2)])/(window*window)
        var = ((s)- (pow((m), 2)))/(window*window)
        std = sqrt(abs(var))
        T = m * (1 + (k * ((std / r) - 1)))
        threshold.append(T)



def transition():
    global value
    for i in range(0 , w-10, 1):
        for j in range(0 , h-10, 1):
            #for m in range(i-5,i+5,1):
                #for l in range(j-5,j+5,1):
                 #   value.append(b[l,m])

            # std2 = pstdev([b[j-1,i-1],b[j-1,i],b[j-1,i+1],b[j,i-1],b[j,i],b[j,i+1],b[j+1,i-1],b[j+1,i],b[j+1,i+1]])
            # m = mean([b[j-1,i-1],b[j-1,i],b[j-1,i+1],b[j,i-1],b[j,i],b[j,i+1],b[j+1,i-1],b[j+1,i],b[j+1,i+1]])
            m = mean([b[j,i],b[j,i-4],b[j,i-3],b[j,i-2],b[j,i-1],b[j,i+1],b[j,i+2],b[j,i+3],b[j,i+4],b[j-4,i-4],b[j-4,i-3],b[j-4,i-2],b[j-4,i-1],b[j-4,i],b[j-4,i+1],b[j-4,i+2],b[j-4,i+3],b[j-4,i+4],b[j-3,i-4],b[j-3,i-3],b[j-3,i-2],b[j-3,i-1],b[j-3,i],b[j-3,i+1],b[j-3,i+2],b[j-3,i+3],b[j-3,i+4],b[j-2,i-4],b[j-2,i-3],b[j-2,i-2],b[j-2,i-1],b[j-2,i],b[j-2,i+1],b[j-2,i+2],b[j-2,i+3],b[j-2,i+4],b[j-1,i-4],b[j-1,i-3],b[j-1,i-2],b[j-1,i-1],b[j-1,i],b[j-1,i+1],b[j-1,i+2],b[j-1,i+3],b[j-1,i+4],b[j+1,i-4],b[j+1,i-3],b[j+1,i-2],b[j+1,i-1],b[j+1,i],b[j+1,i+1],b[j+1,i+2],b[j+1,i+3],b[j+1,i+4],b[j+2,i-4],b[j+2,i-3],b[j+2,i-2],b[j+2,i-1],b[j+2,i],b[j+2,i+1],b[j+2,i+2],b[j+2,i+3],b[j+2,i+4],b[j+3,i-4],b[j+3,i-3],b[j+3,i-2],b[j+3,i-1],b[j+3,i],b[j+3,i+1],b[j+3,i+2],b[j+3,i+3],b[j+3,i+4],b[j+4,i-4],b[j+4,i-3],b[j+4,i-2],b[j+4,i-1],b[j+4,i],b[j+4,i+1],b[j+4,i+2],b[j+4,i+3],b[j+4,i+4]])
            std2=pstdev([b[j,i],b[j,i-4],b[j,i-3],b[j,i-2],b[j,i-1],b[j,i+1],b[j,i+2],b[j,i+3],b[j,i+4],b[j-4,i-4],b[j-4,i-3],b[j-4,i-2],b[j-4,i-1],b[j-4,i],b[j-4,i+1],b[j-4,i+2],b[j-4,i+3],b[j-4,i+4],b[j-3,i-4],b[j-3,i-3],b[j-3,i-2],b[j-3,i-1],b[j-3,i],b[j-3,i+1],b[j-3,i+2],b[j-3,i+3],b[j-3,i+4],b[j-2,i-4],b[j-2,i-3],b[j-2,i-2],b[j-2,i-1],b[j-2,i],b[j-2,i+1],b[j-2,i+2],b[j-2,i+3],b[j-2,i+4],b[j-1,i-4],b[j-1,i-3],b[j-1,i-2],b[j-1,i-1],b[j-1,i],b[j-1,i+1],b[j-1,i+2],b[j-1,i+3],b[j-1,i+4],b[j+1,i-4],b[j+1,i-3],b[j+1,i-2],b[j+1,i-1],b[j+1,i],b[j+1,i+1],b[j+1,i+2],b[j+1,i+3],b[j+1,i+4],b[j+2,i-4],b[j+2,i-3],b[j+2,i-2],b[j+2,i-1],b[j+2,i],b[j+2,i+1],b[j+2,i+2],b[j+2,i+3],b[j+2,i+4],b[j+3,i-4],b[j+3,i-3],b[j+3,i-2],b[j+3,i-1],b[j+3,i],b[j+3,i+1],b[j+3,i+2],b[j+3,i+3],b[j+3,i+4],b[j+4,i-4],b[j+4,i-3],b[j+4,i-2],b[j+4,i-1],b[j+4,i],b[j+4,i+1],b[j+4,i+2],b[j+4,i+3],b[j+4,i+4]])
            T = m * (1 + (k * ((std2 / r)-1 )))
            threshold.append(T)


    return;

def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def pstdev(data):
    """Calculates the population standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/n # the population variance
    return pvar**0.5
# def standard_deviation(lst, population=True):
#     """Calculates the standard deviation for a list of numbers."""
#     num_items = len(lst)
#     mean = sum(lst) / num_items
#     differences = [x - mean for x in lst]
#     sq_differences = [d ** 2 for d in differences]
#     ssd = sum(sq_differences)
#
#     # Note: it would be better to return a value and then print it outside
#     # the function, but this is just a quick way to print out the values along
#     # the way.
#     if population is True:
#         print('This is POPULATION standard deviation.')
#         variance = ssd / num_items
#     else:
#         print('This is SAMPLE standard deviation.')
#         variance = ssd / (num_items - 1)
#     sd = sqrt(variance)
#     # You could `return sd` here.
#
#     print('The mean of {} is {}.'.format(lst, mean))
#     print('The differences are {}.'.format(differences))
#     print('The sum of squared differences is {}.'.format(ssd))
#     print('The variance is {}.'.format(variance))
#     print('The standard deviation is {}.'.format(sd))
#     print('--------------------------')
#transition()

h1,w1=d.shape

for i in range(0, w1, 1):
     for j in range(0, h1, 1):

         if (d[j, i] <= threshold[th]):
             d[j, i] = 0
         else:
             d[j, i] = 255
         th=th+1
for i in range(0, window/2, 1):
    for j in range(0, h1, 1):
        d[j,i]=255
for i in range(0, w1, 1):
    for j in range(0, window/2, 1):
        d[j,i]=255
cv2.imwrite("insta/Sauvola"+str(41)+".jpg", d)
q=q+1;
