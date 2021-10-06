import numpy as np
import cv2 as cv
from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
from matplotlib import pyplot as plt
from tkinter.font import Font

root = Tk()
root.title("Tarea de Programación 01") #Deteccion y emparejamiento de rasgos
#root.geometry("400x400")

def GFTT():
        
        def nothing(x):
                pass

        cv.namedWindow('Frame')
        cv.createTrackbar('r','Frame',1,100,nothing)

        while(1):
                img = cv.imread('blox.jpg')

                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                r = cv.getTrackbarPos('r','Frame')
                corners = cv.goodFeaturesToTrack(gray, r, 0.01, 10)
                corners = np.int0(corners)

                for i in corners:
                        x,y = i.ravel()
                        cv.circle(img,(x,y),3,255,-1)

                cv.imshow('Frame', img)

                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()

def FAST():
        def nothing(x):
                pass

        cv.namedWindow('Frame')
        cv.createTrackbar('threshold','Frame',1,100,nothing)

        while(1):
                img = cv.imread('simple.jpg',0)

                # Initiate FAST object with default values

                threshold = cv.getTrackbarPos('threshold','Frame')
                fast = cv.FastFeatureDetector_create(threshold)

                # find and draw the keypoints
                kp = fast.detect(img,None)
                img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

                # Disable nonmaxSuppression
                fast.setNonmaxSuppression(0)
                kp = fast.detect(img,None)

                img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
                cv.imwrite('fast_false.png',img3)

                cv.imshow("img2",img2)
                cv.imshow('img3', img3)

                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()

def BRIEF():
        img = cv.imread('fly.png', cv.IMREAD_GRAYSCALE)

        # Initiate FAST detector
        star = cv.xfeatures2d.StarDetector_create()

        # Initiate BRIEF extractor
        bytes = int(input('Los valores validos son: 16, 32 y 64: '))
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes)

        # find the keypoints with STAR
        kp = star.detect(img,None)

        # compute the descriptors with BRIEF
        kp, des = brief.compute(img, kp)

        img2 = cv.drawKeypoints(img, kp, None, (255,0,0))
        plt.imshow(img2), plt.show()
        cv.waitKey(0)
        cv.destroyAllWindows()

        
def ORB():
        def nothing(x):
                pass

        cv.namedWindow('Frame')
        cv.createTrackbar('nfeatures','Frame',10,500,nothing)

        while(1):
                img = cv.imread('simple.jpg',0)

                # Initiate ORB detector
                nfeatures = cv.getTrackbarPos('nfeatures','Frame')
                orb = cv.ORB_create(nfeatures)

                # find the keypoints with ORB
                kp = orb.detect(img,None)

                # compute the descriptors with ORB
                kp, des = orb.compute(img, kp)

                # draw only keypoints location,not size and orientation
                img2 = cv.drawKeypoints(img, kp, None, color=(0,0,255), flags=0)
                cv.imwrite('orb.jpg',img2)
                cv.imshow('img2', img2)

                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()

def AGAST():
        def nothing(x):
                pass
        cv.namedWindow('Frame')
        cv.createTrackbar('threshold','Frame',10,100,nothing)
        while(1):
                # Open and convert the input and training-set image from BGR to GRAYSCALE
                img = cv.imread('graf1.png',cv.IMREAD_GRAYSCALE)

                # Initiate AGAST descriptor
                threshold = cv.getTrackbarPos('threshold','Frame')
                AGAST = cv.AgastFeatureDetector_create(threshold)

                # find the keypoints with AGAST
                kp = AGAST.detect(img,None)
                img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

                # Disable nonmaxSuppression
                AGAST.setNonmaxSuppression(0)
                kp = AGAST.detect(img,None)

                # draw only keypoints location,not size and orientation
                img3 = cv.drawKeypoints(img, kp, None, color=(0,0,255), flags=0)
                cv.imwrite('agast_false.png',img3)
                cv.imshow('img3', img3)

                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()

def AKAZE():
        # Open and convert the input and training-set image from BGR to GRAYSCALE
        img = cv.imread('graf1.png',cv.IMREAD_GRAYSCALE)

        # Initiate AKAZE descriptor
        
        AKAZE = cv.AKAZE_create(threshold=0.001)

        # find the keypoints with AKAZE
        kp = AKAZE.detect(img, None)

        # compute the descriptors with AKAZE
        kp, des = AKAZE.compute(img, kp)

        # draw only keypoints location,not size and orientation
        img2 = cv.drawKeypoints(img, kp, None,color=(0,0,255), flags=0)
        plt.imshow(img2), plt.show()

def BRISK():
        def nothing(x):
                pass
        cv.namedWindow('Frame')
        cv.createTrackbar('thresh','Frame',10,255,nothing)

        while(1):
                # Open and convert the input and training-set image from BGR to GRAYSCALE
                img = cv.imread('graf1.png',cv.IMREAD_GRAYSCALE)

                # Initiate BRISK descriptor
                thresh = cv.getTrackbarPos('thresh','Frame')
                BRISK = cv.BRISK_create(thresh)

                # find the keypoints with BRISK
                kp = BRISK.detect(img, None)

                # compute the descriptors with BRISK
                kp, des = BRISK.compute(img, kp)

                # draw only keypoints location,not size and orientation
                img2 = cv.drawKeypoints(img, kp, None, color=(0,0,255), flags=0)
                cv.imwrite('brisk.png',img2)
                cv.imshow('img2', img2)

                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()
        
def KAZE():
        img = cv.imread('graf1.png', cv.IMREAD_GRAYSCALE)

        # Initiate KAZE descriptor
        KAZE = cv.KAZE_create(threshold=0.001)

        # find the keypoints with KAZE
        kp = KAZE.detect(img, None)

        # compute the descriptors with KAZE
        kp, des = KAZE.compute(img, kp)

        # draw only keypoints location,not size and orientation
        img2 = cv.drawKeypoints(img, kp, None, color=(0,0,255), flags=0)
        plt.imshow(img2), plt.show()

def SIFT():
        img = cv.imread('home.jpg')
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        
        sift = cv.SIFT_create()
        kp = sift.detect(gray,None)

        img=cv.drawKeypoints(gray,kp,img, color=(0,0,255), flags=0)
        cv.imwrite('sift_keypoints.jpg',img)
    
        plt.imshow(img), plt.show()
        cv.waitKey(0)
        cv.destroyAllWindows()

def SURF():
        img = cv.imread('fly.png',0)
        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        surf = cv.xfeatures2d.SURF_create(400)

        # Find keypoints and descriptors directly
        kp, des = surf.detectAndCompute(img,None)
        print(len(kp))
 
        # Check present Hessian threshold
        print( surf.getHessianThreshold() )
        400.0
        # We set it to some 50000. Remember, it is just for representing in picture.
        # In actual cases, it is better to have a value 300-500
        surf.setHessianThreshold(50000)
        # Again compute keypoints and check its number.
        kp, des = surf.detectAndCompute(img,None)
        print( len(kp) )

        img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
        plt.imshow(img2),plt.show()
        cv.waitKey(0)
        cv.destroyAllWindows()

def BFSIFT():
        def nothing(x):
                pass

        cv.namedWindow('Frame')
        cv.createTrackbar('match','Frame',10,100,nothing)

        while(1):
                img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
                img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage

                # Initiate SIFT detector
                sift = cv.SIFT_create()

                # find the keypoints and descriptors with SIFT
                kp1, des1 = sift.detectAndCompute(img1,None)
                kp2, des2 = sift.detectAndCompute(img2,None)

                # BFMatcher with default params
                bf = cv.BFMatcher(cv.NORM_L1)
                matches = bf.knnMatch(des1,des2,k=2)

                # Apply ratio test
                good = []
                for m,n in matches:
                        if m.distance < 0.75*n.distance:
                                good.append([m])
                matches = cv.getTrackbarPos('match','Frame')

                # cv.drawMatchesKnn expects list of lists as matches.
                img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good[1:matches],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv.imwrite('bfsift.png',img3)
                cv.imshow('img3', img3)

                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()

def BFSURF():
        # Read Images
        img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
        img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage

        # Initiate SURF detector
        surf = cv.xfeatures2d.SURF_create(400)

        # find the keypoints and descriptors with SURF
        kp1, des1 = surf.detectAndCompute(img1,None)
        kp2, des2 = surf.detectAndCompute(img2,None)

        # BFMatcher with default params
        bf = cv.BFMatcher(cv.NORM_L1)
        matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
                if m.distance < 0.75*n.distance:
                        good.append([m])
        
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.imshow(img3),plt.show()

def BFKAZE():
        def nothing(x):
                pass

        cv.namedWindow('Frame')
        cv.createTrackbar('match','Frame',10,100,nothing)

        while(1):
                # load the image and convert it to grayscale
                img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE)
                img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE)

                # initialize the KAZE descriptor, then detect keypoints and extract
                # local invariant descriptors from the image
                KAZE = cv.KAZE_create(threshold=0.001)
                (kps1, descs1) = KAZE.detectAndCompute(img1, None)
                (kps2, descs2) = KAZE.detectAndCompute(img2, None)

                # Match the features
                bf = cv.BFMatcher(cv.NORM_L1)
                matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed

                # Apply ratio test
                good = []
                for m,n in matches:
                        if m.distance < 0.9*n.distance:
                                good.append([m])
                matches = cv.getTrackbarPos('match','Frame')

                # cv.drawMatchesKnn expects list of lists as matches.
                img3 = cv.drawMatchesKnn(img1, kps1, img2, kps2, good[1:matches], None, flags=2)
                cv.imwrite('bfkaze.png',img3)
                cv.imshow('img3', img3)

                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()

def BFBRIEF():
        def nothing(x):
                pass
        cv.namedWindow('Frame')
        cv.createTrackbar('match','Frame',10,100,nothing)

        while(1):
                img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
                img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage

                # Initiate detector
                star = cv.xfeatures2d.StarDetector_create()

                # Initiate BRIEF detector
                BRIEF = cv.xfeatures2d.BriefDescriptorExtractor_create()

                # find the keypoints with STAR
                kp3 = star.detect(img1,None)
                kp4 = star.detect(img2,None)

                # find the keypoints and descriptors with BRIEF
                kp1, des1 = BRIEF.compute(img1,kp3)
                kp2, des2 = BRIEF.compute(img2,kp4)

                # create BFMatcher object
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

                # Match descriptors.
                matches = bf.match(des1,des2)
                
                # Sort them in the order of their distance.
                match = cv.getTrackbarPos('match','Frame')
                matches = sorted(matches, key = lambda x:x.distance)
                # Draw matches.
                img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:match],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv.imwrite('bfbrief.png',img3)
                cv.imshow('img3', img3)

                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()

def BFBRISK():
        def nothing(x):
                pass
        cv.namedWindow('Frame')
        cv.createTrackbar('thresh','Frame',10,255,nothing)
        cv.createTrackbar('match','Frame',10,100,nothing)
        while(1):
                img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
                img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage
        
                # Initiate BRISK detector
                thresh = cv.getTrackbarPos('thresh','Frame')
                BRISK = cv.BRISK_create(thresh)

                # find the keypoints and descriptors with BRISK
                kp1, des1 = BRISK.detectAndCompute(img1,None)
                kp2, des2 = BRISK.detectAndCompute(img2,None)

                # create BFMatcher object
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

                # Match descriptors.
                matches = bf.match(des1,des2)

                # Sort them in the order of their distance.
                match = cv.getTrackbarPos('match','Frame')
                matches = sorted(matches, key = lambda x:x.distance)

                # Draw first 10 matches.
                img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:match],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv.imwrite('bfbrisk.png',img3)
                cv.imshow('img2', img3)

                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()

def BFORB():
        def nothing(x):
                pass

        cv.namedWindow('Frame')
        cv.createTrackbar('nfeatures','Frame',10,500,nothing)
        cv.createTrackbar('match','Frame',10,100,nothing)
        while(1):
                img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
                img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage

                # Initiate ORB detector
                nfeatures = cv.getTrackbarPos('nfeatures','Frame')
                orb = cv.ORB_create(nfeatures)

                # find the keypoints and descriptors with ORB
                kp1, des1 = orb.detectAndCompute(img1,None)
                kp2, des2 = orb.detectAndCompute(img2,None)

                # create BFMatcher object
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

                # Match descriptors.
                matches = bf.match(des1,des2)

                # Sort them in the order of their distance.
                match = cv.getTrackbarPos('match','Frame')
                matches = sorted(matches, key = lambda x:x.distance)

                # Draw first 10 matches.
                img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:match],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv.imwrite('bforb.png',img3)
                cv.imshow('img3', img3)

                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()

def BFAKAZE():
        def nothing(x):
                pass
        cv.namedWindow('Frame')
        cv.createTrackbar('match','Frame',10,100,nothing)
        while(1):
                # load the image and convert it to grayscale
                img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)
                img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE)

                # initialize the AKAZE descriptor, then detect keypoints and extract
                # local invariant descriptors from the image
                AKAZE = cv.AKAZE_create(threshold=0.001)
                (kps1, descs1) = AKAZE.detectAndCompute(img1, None)
                (kps2, descs2) = AKAZE.detectAndCompute(img2, None)

                # Match the features
                bf = cv.BFMatcher(cv.NORM_HAMMING)
                matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed

                # Apply ratio test
                
                good = []
                for m,n in matches:
                        if m.distance < 0.9*n.distance:
                                good.append([m])
                matches = cv.getTrackbarPos('match','Frame')

                # cv2.drawMatchesKnn expects list of lists as matches.
                img3 = cv.drawMatchesKnn(img1, kps1, img2, kps2, good[1:matches], None, flags=2)
                cv.imwrite('bfakaze.png',img3)
                cv.imshow('img3', img3)

                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()

def FSIFT():
        img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
        img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage

        # Initiate SIFT detector
        sift = cv.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                        matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        plt.imshow(img3,),plt.show()

def FSURF():
        img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
        img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage

        # Initiate SURF detector
        surf = cv.xfeatures2d.SURF_create(400)

        # find the keypoints and descriptors with SURF
        kp1, des1 = surf.detectAndCompute(img1,None)
        kp2, des2 = surf.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

         # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                        matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        plt.imshow(img3,),plt.show()

def FKAZE():
        img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
        img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage

        # Initiate KAZE detector
        kaze = cv.KAZE_create(threshold=0.001)

        # find the keypoints and descriptors with KAZE
        kp1, des1 = kaze.detectAndCompute(img1,None)
        kp2, des2 = kaze.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                        matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        plt.imshow(img3,),plt.show()

def FBRIEF():
        img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
        img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage

        # Initiate FAST detector
        star = cv.xfeatures2d.StarDetector_create()

        # Initiate BRIEF detector
        bytes = int(input('Los valores validos son: 16, 32 y 64: '))
        BRIEF = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes)

         # find the keypoints with STAR
        kp3 = star.detect(img1,None)
        kp4 = star.detect(img2,None)

        # find the keypoints and descriptors with BRIEF
        kp1, des1 = BRIEF.compute(img1,kp3)
        kp2, des2 = BRIEF.compute(img2,kp4)

        # FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
                if m.distance < 0.9*n.distance:
                        matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        plt.imshow(img3,),plt.show()

def FBRISK():
        def nothing(x):
                pass
        
        cv.namedWindow('Frame')
        cv.createTrackbar('thresh','Frame',20,80,nothing)
        while(1):
                img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
                img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage

                # Initiate BRISK detector
                thresh = cv.getTrackbarPos('thresh','Frame')
                BRISK = cv.BRISK_create(thresh)

                # find the keypoints and descriptors with BRISK
                kp1, des1 = BRISK.detectAndCompute(img1,None)
                kp2, des2 = BRISK.detectAndCompute(img2,None)

                # FLANN parameters
                FLANN_INDEX_LSH = 6
                index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
                search_params = dict(checks=50)   # or pass empty dictionary

                flann = cv.FlannBasedMatcher(index_params,search_params)

                matches = flann.knnMatch(des1,des2,k=2)

                # Need to draw only good matches, so create a mask
                matchesMask = [[0,0] for i in range(len(matches))]

                # ratio test as per Lowe's paper
                for i,(m,n) in enumerate(matches):
                        if m.distance < 0.7*n.distance:
                                matchesMask[i]=[1,0]
                
                draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

                img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
                cv.imwrite('fbrisk.png',img3)
                cv.imshow('img3', img3)
                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()

def FORB():
        def nothing(x):
                pass
        
        cv.namedWindow('Frame')
        cv.createTrackbar('nfeatures','Frame',1000,10000,nothing)#minimum value 1000

        while(1):
                img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
                img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage
                # Initiate ORB detector
                nfeatures = cv.getTrackbarPos('nfeatures','Frame')#minimum value 1000
                ORB = cv.ORB_create(nfeatures)

                # find the keypoints and descriptors with ORB
                kp1, des1 = ORB.detectAndCompute(img1,None)
                kp2, des2 = ORB.detectAndCompute(img2,None)

                # FLANN parameters
                FLANN_INDEX_LSH = 6
                index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
                search_params = dict(checks=60)   # or pass empty dictionary

                flann = cv.FlannBasedMatcher(index_params,search_params)

                matches = flann.knnMatch(des1,des2,k=2)

                # Need to draw only good matches, so create a mask
                matchesMask = [[0,0] for i in range(len(matches))]

                # ratio test as per Lowe's paper
                for i,(m,n) in enumerate(matches):
                        if m.distance < 0.75*n.distance:
                                matchesMask[i]=[1,0]

                draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

                img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
                cv.imwrite('forb.png',img3)
                cv.imshow('img3', img3)
                k = cv.waitKey(1) & 0xFF
                if k == 27:
                        break
        cv.destroyAllWindows()

def FAKAZE():
        img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
        img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage

        # Initiate AKAZE detector
        AKAZE = cv.AKAZE_create(threshold=0.001)

        # find the keypoints and descriptors with AKAZE
        kp1, des1 = AKAZE.detectAndCompute(img1,None)
        kp2, des2 = AKAZE.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                        matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        plt.imshow(img3,),plt.show()

font1 = Font(family="Times New Roman", size=12, weight="bold", slant="italic", underline=0, overstrike=0)
font2 = Font(family="Times New Roman", size=12, weight="bold", slant="italic", underline=0, overstrike=0)
font3 = Font(family="Times New Roman", size=12, weight="bold", slant="italic", underline=0, overstrike=0)
font4 = Font(family="Times New Roman", size=12, weight="bold", slant="italic", underline=0, overstrike=0)
font5 = Font(family="Times New Roman", size=12, weight="bold", slant="italic", underline=0, overstrike=0)

my_label_0 = Label(root, text="Seleccione el método a utilizar", font=font1, padx=30, pady=5).grid(row=0, column=1, columnspan=4)
my_label_1 = Label(root, text="Deteccion de rasgos", font=font2, padx=15, pady=5).grid(row=1, column=1)
my_label_2= Label(root, text="Emparejamiento de rasgos", font=font3, padx=30, pady=5).grid(row=1, column=3, columnspan=3)
my_label_3= Label(root, text="Brute Force", font=font4, padx=16, pady=5).grid(row=2, column=3)
my_label_4= Label(root, text="FLANN", font=font5, padx=30, pady=5).grid(row=2, column=4)
my_label_5= Label(root, text="* El método SURF solo se puede usar con versiones anteriores de OpenCV", padx=30, pady=5).grid(row=12, column=1, columnspan=4)

#button_1 = Button(root, text="Deteccion de rasgos", padx=15, pady=5)	
#button_2 = Button(root, text="Emparejamiento de rasgos", padx=30, pady=5)
button_3 = Button(root, text="GFTT", padx=13, pady=5, command=GFTT)
button_4 = Button(root, text="FAST", padx=13, pady=5, command=FAST)
button_5 = Button(root, text="BRIEF", padx=12, pady=5, command=BRIEF)
button_6 = Button(root, text="ORB", padx=15, pady=5, command=ORB)
button_7 = Button(root, text="AGAST", padx=8, pady=5, command=AGAST)
button_8 = Button(root, text="AKAZE", padx=8, pady=5, command=AKAZE)
button_9 = Button(root, text="BRISK", padx=11, pady=5, command=BRISK)
button_10 = Button(root, text="KAZE", padx=12, pady=5, command=KAZE)
button_11 = Button(root, text="SIFT", padx=16, pady=5, command=SIFT)
button_12 = Button(root, text="SURF", padx=13, pady=5, command=DISABLED)
#button_13 = Button(root, text="Brute Force", padx=16, pady=5)
button_14 = Button(root, text="SIFT", padx=16, pady=5, command=BFSIFT)
button_15 = Button(root, text="SURF", padx=13, pady=5, command=DISABLED)
button_16 = Button(root, text="KAZE", padx=13, pady=5, command=BFKAZE)
button_17 = Button(root, text="BRIEF", padx=12, pady=5, command=BFBRIEF)
button_18 = Button(root, text="BRISK", padx=12, pady=5, command=BFBRISK)
button_19 = Button(root, text="ORB", padx=15, pady=5, command=BFORB)
button_20 = Button(root, text="AKAZE", padx=9, pady=5, command=BFAKAZE)
#button_21 = Button(root, text="FLANN", padx=30, pady=5)
button_22 = Button(root, text="SIFT", padx=16, pady=5, command=FSIFT)
button_23 = Button(root, text="SURF", padx=13, pady=5, command=DISABLED)
button_24 = Button(root, text="KAZE", padx=13, pady=5, command=FKAZE)
button_25 = Button(root, text="BRIEF", padx=12, pady=5, command=FBRIEF)
button_26 = Button(root, text="BRISK", padx=12, pady=5, command=FBRISK)
button_27 = Button(root, text="ORB", padx=15, pady=5, command=FORB)
button_28 = Button(root, text="AKAZE", padx=9, pady=5, command=FAKAZE)


#button_1.grid(row=1, column=1)
#button_2.grid(row=1, column=3, columnspan=3)
button_3.grid(row=2, column=1)
button_4.grid(row=3, column=1)
button_5.grid(row=4, column=1)
button_6.grid(row=5, column=1)
button_7.grid(row=6, column=1)
button_8.grid(row=7, column=1)
button_9.grid(row=8, column=1)
button_10.grid(row=9, column=1)
button_11.grid(row=10, column=1)
button_12.grid(row=11, column=1)
#button_13.grid(row=2, column=3)
button_14.grid(row=3, column=3)
button_15.grid(row=4, column=3)
button_16.grid(row=5, column=3)
button_17.grid(row=6, column=3)
button_18.grid(row=7, column=3)
button_19.grid(row=8, column=3)
button_20.grid(row=9, column=3)
#button_21.grid(row=2, column=4)
button_22.grid(row=3, column=4)
button_23.grid(row=4, column=4)
button_24.grid(row=5, column=4)
button_25.grid(row=6, column=4)
button_26.grid(row=7, column=4)
button_27.grid(row=8, column=4)
button_28.grid(row=9, column=4)

root.mainloop()
