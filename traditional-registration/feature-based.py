import cv2
import numpy as np
from ransac import ransac

def main(imgname1, imgname2, detect_method, match_method, bi):

    img1 = cv2.imread(imgname1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(imgname2, cv2.IMREAD_GRAYSCALE)

    detector = None
    if detect_method.lower() == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
    elif detect_method.lower() == 'surf':
        detector = cv2.xfeatures2d.SURF_create()
    elif detect_method.lower() == 'orb':
        detector = cv2.ORB_create()
    elif detect_method.lower() == 'brisk':
        detector = cv2.BRISK_create()
    elif detect_method.lower() == 'akaze':
        detector = cv2.AKAZE_create()
    elif detect_method.lower() == 'kaze':
        detector = cv2.KAZE_create()


    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)

    cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255))
    cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))

    matches1, matches2 = None, None
    if match_method.lower() == 'bf':
        bf = cv2.BFMatcher()
        matches1 = bf.knnMatch(des1,des2, k=2)
        if bi == 1:
            matches2 = bf.knnMatch(des2, des1, k=2)
    elif match_method.lower() == 'flann':
        FLANN_INDEX_KDTREE = 0
        if detect_method.lower() in ['sift', 'surf', 'kaze']:
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            index_params = dict(algorithm=6, table_number=6, key_size=12)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches1 = flann.knnMatch(des1,des2,k=2)
        if bi == 1:
            matches2 = flann.knnMatch(des2, des1, k=2)

    good1, good2 = [], []
    good = []
    for m,n in matches1:
        if m.distance < 0.6*n.distance:
            good1.append(m)
    if bi == 1:
        for m, n in matches2:
            if m.distance < 0.6 * n.distance:
                good2.append(m)

        for m in good1:
            for n in good2:
                if m.queryIdx == n.trainIdx and m.trainIdx == n.queryIdx and m.distance == n.distance:
                    good.append([m])
    else:
        good = [[i] for i in good1]
    print("forward match:%d, reverse match:%d, bidirectional result:%d" % (len(good1), len(good2), len(good)))
    Max_num, good_F, inlier_points = ransac(good, kp1, kp2, confidence=30, iter_num=500)

    img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2, inlier_points,None,flags=2)
    cv2.imshow("match_result", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path1 = '../image/degrade.jpg'
    path2 = '../image/label.jpg'

    '''
        para1: image1 path
        para2: image2 path
        para3: feature detect method, include sift, surf, orb
        para4: descriptor match method, include flann, bf
        para5: whether use bi-directional match method, use:1, not use:0
    '''
    main(path1, path2, 'sift', 'flann', 0)