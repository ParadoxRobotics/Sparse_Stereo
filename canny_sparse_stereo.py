import numpy
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# point cloud declaration
X, Y, Z = [], [], []
# triangulation constant baseline*focal
triangulation_constant = 2945.377*178.232
# Initiate FAST object with default values
orb = cv2.ORB_create(nfeatures=5000)
# Init feature matcher
matcher = cv2.DescriptorMatcher_create("BruteForce")
# import image
left = cv2.imread("im0.png", cv2.IMREAD_GRAYSCALE)
left = cv2.resize(left,(1280,800))
right = cv2.imread("im1.png", cv2.IMREAD_GRAYSCALE)
right = cv2.resize(right,(1280,800))

left_canny = cv2.Canny(left,50,50)
right_canny = cv2.Canny(right,50,50)

# find keypoints
kp_left, d_left = orb.detectAndCompute(left_canny, None)
kp_right, d_right = orb.detectAndCompute(right_canny, None)


# make match
matches = matcher.match(d_left, d_right)
# filter match using SAD
good_match = [m for m in matches if abs(kp_left[m.queryIdx].pt[1] - kp_right[m.trainIdx].pt[1])<2]
print(good_match)
# print matches
img3 = cv2.drawMatches(left, kp_left, right, kp_right, good_match, None, flags=2)
plt.imshow(left_canny)

# calculate disparity with good match
for m in good_match:
    left_pt = kp_left[m.queryIdx].pt
    right_pt = kp_right[m.trainIdx].pt
    dispartity = abs(left_pt[0] - right_pt[0])
    d = triangulation_constant / dispartity
    X.append(left_pt[0])
    Y.append(-left_pt[1])
    Z.append(d)

fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(X, Z, Y)
plt.show()
