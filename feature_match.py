import numpy
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# point cloud declaration
X, Y, Z = [], [], []
# triangulation constant baseline*focal
triangulation_constant = 2945.377*178.232
# Initiate FAST detector with default values
fast = cv2.FastFeatureDetector_create()
# Initiate orb descriptor with default values
orb = cv2.cv2.ORB_create()
# Init feature matcher
matcher = cv2.DescriptorMatcher_create("BruteForce-L1")
# import image
left = cv2.imread("C:/Users/quentin.munch/Desktop/Sparse_Stereo-master/im0.png", cv2.IMREAD_GRAYSCALE)
left = cv2.resize(left,(1280,800))
right = cv2.imread("C:/Users/quentin.munch/Desktop/Sparse_Stereo-master/im1.png", cv2.IMREAD_GRAYSCALE)
right = cv2.resize(right,(1280,800))

# find keypoints using FAST
kp_left = fast.detect(left, None)
kp_right = fast.detect(right, None)
# compute keypoints using ORB descriptor
kp_left_d, d_left = orb.compute(left, kp_left)
kp_right_d, d_right = orb.compute(right, kp_right)

# make match
matches = matcher.match(d_left, d_right)
# filter match using SAD
good_match = [m for m in matches if abs(kp_left_d[m.queryIdx].pt[1] - kp_right_d[m.trainIdx].pt[1])<2]
# print matches
img3 = cv2.drawMatches(left, kp_left_d, right, kp_right_d, good_match, None, flags=2)
plt.imshow(img3)

# calculate disparity with good match
for m in good_match:
    left_pt = kp_left_d[m.queryIdx].pt
    right_pt = kp_right_d[m.trainIdx].pt
    dispartity = abs(left_pt[0] - right_pt[0])
    d = triangulation_constant / dispartity
    X.append(left_pt[0])
    Y.append(-left_pt[1])
    Z.append(d)

fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(X, Z, Y)
plt.show()
