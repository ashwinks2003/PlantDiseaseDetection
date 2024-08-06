import numpy as np
from sklearn import svm
import csv
import argparse
import cv2
from skimage.feature import greycomatrix, greycoprops

# Parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('input_img', help='the input image file')
args = parser.parse_args()

# Reading the input image
image = cv2.imread(args.input_img)
f1 = 0
f2 = 0
f3 = 0
f4 = 0

# Feature 1: Amount of green color in the picture.
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
boundaries = [([30, 0, 0], [70, 255, 255])]
for (lower, upper) in boundaries:
    mask = cv2.inRange(hsv, (36, 0, 0), (70, 255, 255))
    ratio_green = cv2.countNonZero(mask) / (image.size / 3)
    f1 = np.round(ratio_green, 2)

# Feature 2: Amount of non-green color in the picture
f2 = 1 - f1

# Feature 3: Periphery length
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv = cv2.split(hsv)
gray = hsv[0]
gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=-1, sigmaY=-1)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (255, 0, 0), thickness=2)
perimeter = 0
for c in contours:
    perimeter += cv2.arcLength(c, True)
f3 = perimeter / 15000

# Feature 4: Contrast
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
g = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
contrast = greycoprops(g, 'contrast')
f4 = contrast[0][0] + contrast[0][1] + contrast[0][2] + contrast[0][3]
f4 = f4 / 2000000000

# Reading CSV file
filename = "out.csv"
fields = []
rows = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)
    print("Total no. of rows: %d" % (csvreader.line_num))

print('Field names are:' + ', '.join(field for field in fields))

# Preparing data for SVM
A = []
Y = []
included = [3, 4, 5, 6]
y1 = [2]
for row in rows:
    content = list(float(row[i]) for i in included)
    A.append(content)
for row in rows:
    content = list(float(row[i]) for i in y1)
    Y.append(content)

b = np.reshape(Y, (len(Y),))
print(b.shape)
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(A, b)
print(clf.predict([[f1, f2, f3, f4]]))
print(f1)
print(f2)
print(f3)
print(f4)
