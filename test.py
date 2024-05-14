
# Important NOTE:  Use opencv >=4.4 
import cv2
 
# Loading the image
img = cv2.imread('./data/nerf/chair_sim_depth/images/chair_2_20.png')
 
 # Converting image to grayscale
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
# Applying SIFT detector
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)
 
# Marking the keypoint on the image using circles
img=cv2.drawKeypoints(gray ,
                      kp ,
                      img ,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(kp[0].pt)
cv2.imwrite('image-with-keypoints.jpg', img)
