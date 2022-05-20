# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import cv2
import os
import glob
import numpy as np

def CameraCalibrate(x):
# Defining the dimensions of checkerboard
    CHECKERBOARD = (7,7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
     
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 


    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob(x)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If desired number of corner are detected, we refine the pixel coordinates and display them on the images of checker board
        print(ret)
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
            imgpoints.append(corners2)
            
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
    
        cv2.imshow('img',img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    h,w = img.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    # Performing camera calibration by passing the value of known 3D points (objpoints) and corresponding pixel coordinates of the detected corners (imgpoints)

    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)

    R = cv2.Rodrigues(rvecs[0])[0]
    t = tvecs[0]
    Rt = np.concatenate([R,t], axis=-1) # [R|t]
    P = np.matmul(mtx,Rt) # A[R|t]
    return P,objpoints,imgpoints,mtx,gray,dist

# Returns the projection matrix for a given set of corresponding 2D and
# 3D points. 
# 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
# 'Points_3D' is nx3 matrix of 3D coordinate of points in the world
# 'M' is the 3x4 projection matrix
def calculate_projection_matrix(Points_2D, Points_3D):
    # To solve for the projection matrix. You need to set up a system of
    # equations using the corresponding 2D and 3D points:
    #
    #                                                     [M11       [ u1
    #                                                      M12         v1
    #                                                      M13         .
    #                                                      M14         .
    # [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
    #  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
    #  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
    #  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
    #  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
    #                                                      M32         un
    #                                                      M33         vn ]
    #
    # Then you can solve this using least squares with the 'np.linalg.lstsq' operator.
    # Notice you obtain 2 equations for each corresponding 2D and 3D point
    # pair. To solve this, you need at least 6 point pairs. Note that we set
    # M34 = 1 in this scenario. If you instead choose to use SVD via np.linalg.svd, you should
    # not make this assumption and set up your matrices by following the 
    # set of equations on the project page. 
    #
    ##################
    # Your code here #
    ##################
    matrixA = []
    matrixB = []
    rows, cols = Points_2D.shape

    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    # Your total residual should be less than 1.
    #print('Randomly setting matrix entries as a placeholder')
    #M = np.array([[0.1768, 0.7018, 0.7948, 0.4613],
                  #[0.6750, 0.3152, 0.1136, 0.0480],
                  #[0.1020, 0.1725, 0.7244, 0.9932]])
    
    
    for x in range(0, rows): #iterate at each point across the 20 row
        matrixA.append([Points_3D[x, 0] ,Points_3D[x, 1], Points_3D[x, 2], 1, 0, 0, 0, 0 , -Points_2D[x, 0]*Points_3D[x, 0], -Points_2D[x, 0]*Points_3D[x, 1], -Points_2D[x, 0]*Points_3D[x, 2]])
        matrixA.append([0, 0, 0, 0, Points_3D[x, 0], Points_3D[x, 1], Points_3D[x, 2], 1, -Points_2D[x, 1]*Points_3D[x, 0], -Points_2D[x, 1]*Points_3D[x, 1], -Points_2D[x, 1]*Points_3D[x, 2]])
        matrixB.append([Points_2D[x, 0]])
        matrixB.append([Points_2D[x, 1]])
    # B = A * M
    # multiplying by A_transpose for both sides
    AT_A = np.dot(np.mat(matrixA).T, np.mat(matrixA))
    AT_B = np.dot(np.mat(matrixA).T, np.mat(matrixB)) # A_transpose * B = A_transpose * A * M
    # M = Inverse(A_transpose * A) * (A_transpose * B)
    rhs = np.linalg.inv(AT_A)
    lhs = AT_B
    M = np.reshape(np.append(np.array( np.dot(rhs, lhs).T),[1]),(3,4)) #Final projection matrix

    return M


# Returns the camera center matrix for a given projection matrix
# 'M' is the 3x4 projection matrix
# 'Center' is the 1x3 matrix of camera center location in world coordinates
def compute_camera_center(M):
    ##################
    # Your code here #
    ##################

    # Replace this with the correct code
    # In the visualization you will see that this camera location is clearly
    # incorrect, placing it in the center of the room where it would not see all
    # of the points.
    #Center = np.array([1, 1, 1])
    Center = np.dot(np.linalg.inv(np.dot(-M[:,0:3].T, -M[:,0:3])), np.dot(-M[:,0:3].T, M[:,3])) #C= -Q^-1 * m4
    return Center
