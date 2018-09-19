# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================
import sys
import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region

    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h,w), dtype=np.uint16)

    pattern001 = cv2.resize(cv2.imread("images/pattern001.jpg") / 255.0, (0,0), fx=scale_factor,fy=scale_factor)

    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        # print("images/pattern%03d.jpg")
        patt = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2)), (0,0), fx=scale_factor,fy=scale_factor)
        patt_gray = cv2.cvtColor(patt, cv2.COLOR_BGR2GRAY)/255.0
        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)

        # TODO: populate scan_bits by putting the bit_code according to on_mask
        on_mask2 = on_mask * bit_code
        scan_bits += on_mask2
    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    correspondences = np.zeros((h,w, 3))
    color = []
    camera_points = []
    projector_points = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code

            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points

            p_x, p_y = binary_codes_ids_codebook[ scan_bits[y,x] ]

            if p_x >= 1279 or p_y >= 799: # filter
                continue

            camera_points.append(np.array([x/2.0, y/2.0]))
            projector_points.append([p_x,p_y])
            # add correspondence
            correspondences[y, x, 2] = p_x
            correspondences[y, x, 1] = p_y
            # add color for bonus
            color.append(pattern001[y,x,:])

    # write the correspondence image
    correspondences[:, :, 2] /= np.max(correspondences[:,:,2])
    correspondences[:, :, 2] *= 255.0

    correspondences[:, :, 1] /= np.max(correspondences[:,:,1])
    correspondences[:, :, 1] *= 255.0

    cv2.imwrite("correspondence.jpg", correspondences)

    # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # R1,R1,P1,P2,Q,_,_ = cv2.stereoRectify(camera_K,camera_d,projector_K,projector_d,img_size,projector_R,projector_t)

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    # cv2.undistortPoints(src, K, dist_coef)
    length = len(camera_points)
    camera_points = np.asarray(camera_points)
    camera_points = np.reshape(camera_points,(length,1,2))
    camera_points = np.float32(camera_points)
    norm_camera_points = cv2.undistortPoints(camera_points,camera_K,camera_d)

    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d (api change: remove dst )
    projector_length = len(projector_points)
    projector_points = np.asarray(projector_points)
    projector_points = np.reshape(projector_points,(projector_length,1,2))
    projector_points = np.float32(projector_points)
    norm_projector_points = cv2.undistortPoints(projector_points,projector_K,projector_d)

    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    camera_R_t = np.hstack((np.identity(3),np.zeros((3,1))))
    P1 = camera_R_t

    projector_R_t = np.hstack((projector_R,projector_t))
    P2 = projector_R_t

    points4D = cv2.triangulatePoints(P1, P2, norm_camera_points,norm_projector_points)

    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    points_3d = cv2.convertPointsFromHomogeneous(points4D.T)

    mask = (points_3d[:,:,2] > 200) & (points_3d[:,:,2] < 1400)
    points_3d_back = []

    for i in range(len(points_3d)):
        points_3d[i] = points_3d[i]*mask[i]
        if mask[i] == True:
          points_3d_back.append(points_3d[i])


    # TODO: name the resulted 3D points as "points_3d"
    points_3d = points_3d_back

    # Bonus output_color.xyzrgb
    output_name = sys.argv[1] + "output_color.xyzrgb"
    with open(output_name,"w") as f:
        for i in range(len(points_3d)):
            p = points_3d[i]
            c = color[i]
            f.write("%d %d %d %d %d %d \n"%(p[0,0],p[0,1],p[0,2],c[0],c[1],c[2]))

    return points_3d

def write_3d_points(points_3d):

	# ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))
    return points_3d


if __name__ == '__main__':

	# ===== DO NOT CHANGE THIS FUNCTION =====

	# validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
