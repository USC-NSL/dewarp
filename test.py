from SimpleCV import Camera, VideoStream, Color, Display, Image, VirtualCamera
import cv2
import numpy as np
import time
from picamera import PiCamera


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
# Load the video from the rpi
# vc = VirtualCamera("video.h264","video")
camera = PiCamera()
camera.start_preview()
for i in range(5):
    sleep(5)
    camera.capture('./image%s.jpg' % i)
    img = cv2.imread('./image%s.jpg' % i)
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)

    cv2.imshow('dst_rt', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
