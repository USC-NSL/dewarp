from SimpleCV import Camera, VideoStream, Color, Display, Image, VirtualCamera
import cv2
import numpy as np
import time
from picamera import PiCamera

import matplotlib.pyplot as plt

# deprecated, checks if point in the sphere is in our output
def isInROI(x,y,R1,R2,Cx,Cy):
    isInOuter = False
    isInInner = False
    xv = x-Cx
    yv = y-Cy
    rt = (xv*xv)+(yv*yv)
    if( rt < R2*R2 ):
        isInOuter = True
        if( rt < R1*R1 ):
            isInInner = True
    return isInOuter and not isInInner
# build the mapping
def buildMap(Ws,Hs,Wd,Hd,R1,R2,Cx,Cy):
    map_x = np.zeros((Hd,Wd),np.float32)
    map_y = np.zeros((Hd,Wd),np.float32)
    for y in range(0,int(Hd-1)):
        for x in range(0,int(Wd-1)):
            r = (float(y)/float(Hd))*(R2-R1)+R1
            theta = (float(x)/float(Wd))*2.0*np.pi
            xS = Cx+r*np.sin(theta)
            yS = Cy+r*np.cos(theta)
            map_x.itemset((y,x),int(xS))
            map_y.itemset((y,x),int(yS))
        
    return map_x, map_y
# do the unwarping 
def unwarp(img,xmap,ymap):
    output = cv2.remap(img.getNumpyCv2(),xmap,ymap,cv2.INTER_LINEAR)
    result = Image(output,cv2image=True)
    return result


disp = Display((800,600))
vals = []
last = (0,0)
i = 0

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global vals, cropping
 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        vals = [(x, y)]
        cropping = True
 
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        vals.append((x, y))
        cropping = False


# Load the video from the rpi
# vc = VirtualCamera("video.h264","video")
camera = PiCamera()


camera.capture('./image%s.jpg' % i)
img = cv2.imread('./image%s.jpg' % i)

clone = img.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
 
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        img = clone.copy()
 
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

cv2.destroyAllWindows()



# 0 = xc yc
# 1 = r1
# 2 = r2
# center of the "donut"    
Cx = vals[0][0]
Cy = vals[0][1]
# Inner donut radius
R1x = vals[1][0]
R1y = vals[1][1]
R1 = np.sqrt((R1x-Cx)**2 + (R1y-Cy)**2)
# outer donut radius
R2x = vals[2][0]
R2y = vals[2][1]
R2 = np.sqrt((R2x-Cx)**2 + (R2y-Cy)**2)

Wd = int(2.0*((R2+R1)/2)*np.pi)
Hd = int(R2-R1)
Ws = img.width
Hs = img.height
# build the pixel map, this could be sped up
print "BUILDING MAP!"
xmap,ymap = buildMap(Ws,Hs,Wd,Hd,R1,R2,Cx,Cy)
print "MAP DONE!"

for i in range(5):
    time.sleep(5)
    camera.capture('./image%s.jpg' % i)
    img = cv2.imread('./image%s.jpg' % i)
    result = unwarp(img,xmap,ymap)

    screen_res = 1280, 720
    scale_width = screen_res[0] / result.shape[1]
    scale_height = screen_res[1] / result.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(result.shape[1] * scale)
    window_height = int(result.shape[0] * scale)
    cv2.imwrite('./image%s.jpg' % i, img)

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)

    cv2.imshow('dst_rt', result)
    # cv2.waitKey(0)
    

cv2.destroyAllWindows()
