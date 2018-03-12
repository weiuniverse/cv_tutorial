# Instructions:
# Do not change the output file names, use the helper functions as you see fit

import os
import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Perspective warping")
    print("2 Cylindrical warping")
    print("3 Bonus perspective warping")
    print("4 Bonus cylindrical warping")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image1] " + "[path to input image2] " + "[path to input image3] " + "[output directory]")

'''
Detect, extract and match features between img1 and img2.
Using SIFT as the detector/extractor, but this is inconsequential to the user.

Returns: (pts1, pts2), where ptsN are points on image N.
    The lists are "aligned", i.e. point i in pts1 matches with point i in pts2.

Usage example:
    img1 = cv2.imread("image1.jpg", 0)
    img2 = cv2.imread("image2.jpg", 0)
    (pts1, pts2) = feature_matching(img1, img2)

    plt.subplot(121)
    plt.imshow(img1)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
    plt.subplot(122)
    plt.imshow(img2)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
'''
def feature_matching(img1, img2, savefig=False):
    '''
    Detect, extract and match features between img1 and img2.
    Using SIFT as the detector/extractor, but this is inconsequential to the user.

    Returns: (pts1, pts2), where ptsN are points on image N.
        The lists are "aligned", i.e. point i in pts1 matches with point i in pts2.

    Usage example:
        img1 = cv2.imread("image1.jpg", 0)
        img2 = cv2.imread("image2.jpg", 0)
        (pts1, pts2) = feature_matching(img1, img2)

        plt.subplot(121)
        plt.imshow(img1)
        plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
        plt.subplot(122)
        plt.imshow(img2)
        plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
    '''
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)

    matchesMask_ratio = [[0,0] for i in xrange(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in xrange(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]

    if savefig:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask_ratio_recip,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        plt.figure(),plt.xticks([]),plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])

'''
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching

Usage example:

    im = cv2.imread("myimage.jpg",0) #grayscale
    h,w = im.shape
    f = 700
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)
'''
def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0,0]

    im_h,im_w = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h,cyl_w = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0,cyl_w):
        for y_cyl in np.arange(0,cyl_h):
            theta = (x_cyl - x_c) / f
            h     = (y_cyl - y_c) / f
            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K,X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
            cyl_mask[int(y_cyl),int(x_cyl)] = 255


    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png",bbox_inches='tight')

    return (cyl,cyl_mask)

'''
Calculate the geometric transform (only affine or homography) between two images,
based on feature matching and alignment with a robust estimator (RANSAC).

Returns: (M, pts1, pts2, mask)
Where: M    is the 3x3 transform matrix
       pts1 are the matched feature points in image 1
       pts2 are the matched feature points in image 2
       mask is a binary mask over the lists of points that selects the transformation inliers

Usage example:
    img1 = cv2.imread("image1.jpg", 0)
    img2 = cv2.imread("image2.jpg", 0)
    (M, pts1, pts2, mask) = getTransform(img1, img2)

    # for example: transform img1 to img2's plane
    # first, make some room around img2
    img2 = cv2.copyMakeBorder(img2,200,200,500,500, cv2.BORDER_CONSTANT)
    # then transform img1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(img1, M, (img2.shape[1],img2.shape[0]), dst=img2.copy(), borderMode=cv2.BORDER_TRANSPARENT)

    plt.imshow(out, cmap='gray')
    plt.show()
'''
def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        M = np.append(M, [[0,0,1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)

# ===================================================
# ================ Perspective Warping ==============
# ===================================================
def Perspective_warping(img1, img2, img3):

	# Write your codes here

    # first, make some room around img1
    img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)

    # then transform img2,3 with the 3x3 transformation matrix to img1 plane
    (M21, pts2, pts1, mask) = getTransform(img2, img1,method='homography')
    out = cv2.warpPerspective(img2, M21, (img1.shape[1],img1.shape[0]), dst=img1.copy(),flags = cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)

    (M31, pts3, pts1, mask) = getTransform(img3, img1,method='homography')
    output_image = cv2.warpPerspective(img3, M31, (img1.shape[1],img1.shape[0]), dst=out,flags = cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)

	# Write out the result
    output_name = sys.argv[5] + "output_homography.png"
    cv2.imwrite(output_name, output_image)

    return True

def Bonus_perspective_warping(img1, img2, img3):

	# Write your codes here
    # first, make some room around img1
    img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)
    bg = img1.copy()
    bg = cv2.bitwise_and(bg,0)
    rows,cols = img1.shape
    # then transform img2,3 with the 3x3 transformation matrix to img1 plane
    (M21, pts2, pts1, mask) = getTransform(img2,img1,method='homography')
    out2 = cv2.warpPerspective(img2, M21, (img1.shape[1],img1.shape[0]),dst=bg.copy(),flags = cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
    rows,cols = out2.shape
    mask2 = out2.copy()
    for i in range(rows):
        for j in range(cols):
            if mask2[i][j] <> 0:
                mask2[i][j] = 1

    num_levels = 4
    GA = out2.copy()
    GB = img1.copy()
    GM = mask2.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in xrange(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in xrange(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        # LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LA = cv2.pyrUp(gpA[i])
        LA = np.resize(LA, gpA[i-1].shape)
        LA = np.subtract(gpA[i-1], LA)

        LB = cv2.pyrUp(gpB[i])
        LB = np.resize(LB, gpB[i-1].shape)
        LB = np.subtract(gpB[i-1], LB)

        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = np.resize(ls_, LS[i].shape)
        ls_ = cv2.add(np.float32(ls_), np.float32(LS[i]))

    out12 = ls_
    mask12 = out12.copy()
    for i in range(rows):
        for j in range(cols):
            if mask12[i][j] <> 0:
                mask12[i][j] = 1

    (M31, pts1, pts2, mask) = getTransform(img3,img1,method='homography')
    out3 = cv2.warpPerspective(img3, M31, (img1.shape[1],img1.shape[0]),dst=bg.copy(),flags = cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
    mask3 = out3.copy()
    for i in range(rows):
        for j in range(cols):
            if mask3[i][j] <> 0:
                mask3[i][j] = 1
    GA = out3.copy()
    GB = out12.copy()
    GM = mask3.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in xrange(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in xrange(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        # LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LA = cv2.pyrUp(gpA[i])
        LA = np.resize(LA, gpA[i-1].shape)
        LA = np.subtract(gpA[i-1], LA)

        LB = cv2.pyrUp(gpB[i])
        LB = np.resize(LB, gpB[i-1].shape)
        LB = np.subtract(gpB[i-1], LB)

        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = np.resize(ls_, LS[i].shape)
        ls_ = cv2.add(np.float32(ls_), np.float32(LS[i]))

    output_image = ls_ # This is dummy output, change it to your output

	# Write out the result
    output_name = sys.argv[5] + "output_homography_lpb.png"
    cv2.imwrite(output_name, output_image)

    return True

# ===================================================
# =============== Cynlindrical Warping ==============
# ===================================================
def Cylindrical_warping(img1, img2, img3):

	# Write your codes here
    h1,w1 = img1.shape
    f = 420
    K1 = np.array([[f, 0, w1/2], [0, f, h1/2], [0, 0, 1]]) # mock calibration matrix
    imcyl_1, mask1 = cylindricalWarpImage(img1, K1)

    h2,w2 = img2.shape
    f = 420
    K2 = np.array([[f, 0, w2/2], [0, f, h2/2], [0, 0, 1]]) # mock calibration matrix
    imcyl_2, mask2 = cylindricalWarpImage(img2, K2)

    h3,w3 = img3.shape
    f = 420
    K3 = np.array([[f, 0, w3/2], [0, f, h3/2], [0, 0, 1]]) # mock calibration matrix
    imcyl_3, mask3 = cylindricalWarpImage(img3, K3)

    imcyl_1 = cv2.copyMakeBorder(imcyl_1,50,50,300,300, cv2.BORDER_CONSTANT)

    (M21, pts2, pts1, mask) = getTransform(imcyl_2,imcyl_1,method='affine')
    M21 = M21[:2,:]
    bg = imcyl_1.copy()
    bg = cv2.bitwise_and(bg,0)
    out2 = cv2.warpAffine(imcyl_2, M21, (imcyl_1.shape[1],imcyl_1.shape[0]),dst=bg.copy(),flags = cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    out_mask2 = cv2.warpAffine(mask2, M21, (imcyl_1.shape[1],imcyl_1.shape[0]),dst=bg.copy(),flags = cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
    mask_inv2 = cv2.bitwise_not(out_mask2)
    plane = imcyl_1.copy()
    img1_bg = cv2.bitwise_and(plane,plane,mask = mask_inv2)
    out12 = cv2.add(img1_bg,out2)

    (M31, pts3, pts1, mask) = getTransform(imcyl_3,imcyl_1,method='affine')
    M31 = M31[:2,:]
    out3 = cv2.warpAffine(imcyl_3, M31, (imcyl_1.shape[1],imcyl_1.shape[0]),dst=bg.copy(),flags = cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
    out_mask3 = cv2.warpAffine(mask3, M31, (imcyl_1.shape[1],imcyl_1.shape[0]),dst=bg.copy(),flags = cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
    mask_inv3 = cv2.bitwise_not(out_mask3)
    plane12 = out12.copy()
    img12_bg = cv2.bitwise_and(plane12,plane12,mask = mask_inv3)
    output_image = cv2.add(img12_bg,out3)

	# Write out the result
    output_name = sys.argv[5] + "output_cylindrical.png"
    cv2.imwrite(output_name, output_image)

    return True

def Bonus_cylindrical_warping(img1, img2, img3):
	# Write your codes here
    h1,w1 = img1.shape
    f = 500
    K1 = np.array([[f, 0, w1/2], [0, f, h1/2], [0, 0, 1]]) # mock calibration matrix
    imcyl_1, mask1 = cylindricalWarpImage(img1, K1)

    h2,w2 = img2.shape
    f = 500
    K2 = np.array([[f, 0, w2/2], [0, f, h2/2], [0, 0, 1]]) # mock calibration matrix
    imcyl_2, mask2 = cylindricalWarpImage(img2, K2)

    h3,w3 = img3.shape
    f = 500
    K3 = np.array([[f, 0, w3/2], [0, f, h3/2], [0, 0, 1]]) # mock calibration matrix
    imcyl_3, mask3 = cylindricalWarpImage(img3, K3)

    # first, make some room around img1
    imcyl_1 = cv2.copyMakeBorder(imcyl_1,50,50,300,300, cv2.BORDER_CONSTANT)
    bg = imcyl_1.copy()
    bg = cv2.bitwise_and(bg,0)
    # then transform img2,3 with the 3x3 transformation matrix to img1 plane
    (M21, pts2, pts1, mask) = getTransform(imcyl_2,imcyl_1,method='affine')
    M21 = M21[:2,:]
    out2 = cv2.warpAffine(imcyl_2, M21, (imcyl_1.shape[1],imcyl_1.shape[0]),dst=bg.copy(),flags = cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    out_mask2 = cv2.warpAffine(mask2, M21, (imcyl_1.shape[1],imcyl_1.shape[0]),dst=bg.copy(),flags = cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
    out_mask2 = out_mask2/255.0

    num_levels = 4
    GA = out2.copy()
    GB = imcyl_1.copy()
    GM = out_mask2.copy()

    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in xrange(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in xrange(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = cv2.pyrUp(gpA[i])
        LA = np.resize(LA, gpA[i-1].shape)
        LA = np.subtract(gpA[i-1], LA)

        LB = cv2.pyrUp(gpB[i])
        LB = np.resize(LB, gpB[i-1].shape)
        LB = np.subtract(gpB[i-1], LB)

        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = np.resize(ls_, LS[i].shape)
        ls_ = cv2.add(np.float32(ls_), np.float32(LS[i]))

    out12 = ls_


    (M31, pts3, pts1, mask) = getTransform(imcyl_3,imcyl_1,method='affine')
    empty = 0
    M31 = M31[:2,:]
    plane12 = out12.copy()
    out3 = cv2.warpAffine(imcyl_3, M31, (imcyl_1.shape[1],imcyl_1.shape[0]),dst=bg.copy(),flags = cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
    out_mask3 = cv2.warpAffine(mask3, M31, (imcyl_1.shape[1],imcyl_1.shape[0]),dst=bg.copy(),flags = cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
    out_mask3 = out_mask3/255.0


    GA = out3.copy()
    GB = out12.copy()
    GM = out_mask3.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in xrange(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in xrange(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        # LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LA = cv2.pyrUp(gpA[i])
        LA = np.resize(LA, gpA[i-1].shape)
        LA = np.subtract(gpA[i-1], LA)

        LB = cv2.pyrUp(gpB[i])
        LB = np.resize(LB, gpB[i-1].shape)
        LB = np.subtract(gpB[i-1], LB)

        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)



    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = np.resize(ls_, LS[i].shape)
        ls_ = cv2.add(np.float32(ls_), np.float32(LS[i]))

	output_image = ls_ # This is dummy output, change it to your output

	# Write out the result
    output_name = sys.argv[5] + "output_cylindrical_lpb.png"
    cv2.imwrite(output_name, output_image)
    return True

'''
This exact function will be used to evaluate your results for HW2
Compare your result with master image and get the difference, the grading
criteria is posted on Piazza
'''
def RMSD(target, master):
    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        return -1
    else:
        total_diff = 0.0;
        master_channels = cv2.split(master);
        target_channels = cv2.split(target);

        for i in range(0, len(master_channels), 1):
            dst = cv2.absdiff(master_channels[i], target_channels[i])
            dst = cv2.pow(dst, 2)
            mean = cv2.mean(dst)
            total_diff = total_diff + mean[0]**(1/2.0)

    return total_diff;

if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) != 6):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])
    if (question_number > 4 or question_number < 1):
        print("Input parameters out of bound ...")
        sys.exit()

    input_image1 = cv2.imread(sys.argv[2], 0)
    input_image2 = cv2.imread(sys.argv[3], 0)
    input_image3 = cv2.imread(sys.argv[4], 0)

    function_launch = {
    1 : Perspective_warping,
    2 : Cylindrical_warping,
    3 : Bonus_perspective_warping,
    4 : Bonus_cylindrical_warping,
    }
    # Call the function
    function_launch[question_number](input_image1, input_image2, input_image3)
