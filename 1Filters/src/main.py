# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output

# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):

   # Write histogram equalization here

   b,g,r=cv2.split(img_in)

   hist = cv2.calcHist([b],[0],None,[256],[0,256])
   hist = hist[:,0]
   cdf = numpy.cumsum(hist)
   cdf_normalized = numpy.uint8(cdf * 256.0 / (cdf.max()+1))
   b = cdf_normalized[b]

   hist = cv2.calcHist([g],[0],None,[256],[0,256])
   hist = hist[:,0]
   cdf = numpy.cumsum(hist)
   cdf_normalized = numpy.uint8(cdf * 256.0 / (cdf.max()+1))
   g = cdf_normalized[g]

   hist = cv2.calcHist([r],[0],None,[256],[0,256])
   hist = hist[:,0]
   cdf = numpy.cumsum(hist)
   cdf_normalized = numpy.uint8(cdf * 256.0 / (cdf.max()+1))
   r = cdf_normalized[r]

   img_in = cv2.merge([b,g,r])
   img_out = img_in # Histogram equalization result

   return True, img_out

def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);

   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)

   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):

   # Write low pass filter here
   img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
   dft = cv2.dft(numpy.float32(img_in),flags = cv2.DFT_COMPLEX_OUTPUT)
   dft_shift = numpy.fft.fftshift(dft)
   rows,cols = img_in.shape
   crow = rows/2
   ccol = cols/2
   mask = numpy.zeros((rows,cols,2),numpy.uint8)
   mask[crow-10:crow+10, ccol-10:ccol+10] = 1
   fshift = dft_shift*mask
   f_ishift = numpy.fft.ifftshift(fshift)
   img_back = cv2.idft(f_ishift)
   img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
   img_in= (img_back-numpy.amin(img_back))/(numpy.amax(img_back)-numpy.amin(img_back))
   img_in = numpy.uint8(img_in*256)

   img_out = img_in # Low pass filter result

   return True, img_out

def high_pass_filter(img_in):

   # Write high pass filter here
   img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
   dft = cv2.dft(numpy.float32(img_in),flags = cv2.DFT_COMPLEX_OUTPUT)
   dft_shift = numpy.fft.fftshift(dft)
   rows,cols = img_in.shape
   crow = rows/2
   ccol = cols/2
   # print(crow,ccol)
   dft_shift[crow-10:crow+10, ccol-10:ccol+10] = 0
   dft_ishift = numpy.fft.ifftshift(dft_shift)
   img_back = cv2.idft(dft_ishift)
   img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
   img_in = (img_back-numpy.amin(img_back))/(numpy.amax(img_back)-numpy.amin(img_back))
   img_in = numpy.uint8(img_in*256)

   img_out = img_in # High pass filter result

   return True, img_out

def deconvolution(img_in):

   # Write deconvolution codes here
   gk = cv2.getGaussianKernel(21,5)
   gk = gk * gk.T
   dft = numpy.fft.fft2(numpy.float32(img_in),(img_in.shape[0],img_in.shape[1]))
   imf = numpy.fft.fftshift(dft)
   # imf = ft(img_in, (img_in.shape[1],img_in.shape[1])) # make sure sizes match
   # gkf = ft(gk, (img_in.shape[1],img_in.shape[1])) # so we can multiple easily
   dft = numpy.fft.fft2(numpy.float32(gk),(img_in.shape[0],img_in.shape[1]))
   gkf = numpy.fft.fftshift(dft)
   imdeconv = numpy.true_divide(imf,gkf)
   # img_back = ift(imdeconvf)
   f_ishift = numpy.fft.ifftshift(imdeconv)
   img_back = numpy.fft.ifft2(f_ishift)
   img_back =  numpy.abs(img_back)
   img_in = (img_back-numpy.amin(img_back))/(numpy.amax(img_back)-numpy.amin(img_back))
   img_in = numpy.uint8(img_in*256)

   img_out = img_in # Deconvolution result
   return True, img_out

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread("blurred2.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)

   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)

   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)

   return True

# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   img_in1 = img_in1[:,:img_in1.shape[0]]
   img_in2 = img_in2[:img_in1.shape[0],:img_in1.shape[0]]
   pD1 = img_in1.copy()
   pD2 = img_in2.copy()
   list_pD1 = [pD1]
   list_pD2 = [pD2]
   for i in range(5):
      pD1=cv2.pyrDown(pD1)
      list_pD1.append(pD1)
      pD2=cv2.pyrDown(pD2)
      list_pD2.append(pD2)
   list_lp1 = [list_pD1[4]]
   list_lp2 = [list_pD2[4]]

   for i in range(4,0,-1):
      height,width = list_pD1[i-1].shape[0:2]
      pU1 = cv2.pyrUp(list_pD1[i],dstsize=(width,height))
      lp1 = pU1.copy()
      lp1[:,:,0] = cv2.subtract(list_pD1[i-1][:,:,0],pU1[:,:,0])
      lp1[:,:,1] = cv2.subtract(list_pD1[i-1][:,:,1],pU1[:,:,1])
      lp1[:,:,2] = cv2.subtract(list_pD1[i-1][:,:,2],pU1[:,:,2])
      list_lp1.append(lp1)

      height,width = list_pD2[i-1].shape[0:2]
      pU2 = cv2.pyrUp(list_pD2[i],dstsize=(width,height))#,dstsize=list_pD1[i-1].shape[0:2])
      lp2 = pU2.copy()
      lp2[:,:,0] = cv2.subtract(list_pD2[i-1][:,:,0],pU2[:,:,0])
      lp2[:,:,1] = cv2.subtract(list_pD2[i-1][:,:,1],pU2[:,:,1])
      lp2[:,:,2] = cv2.subtract(list_pD2[i-1][:,:,2],pU2[:,:,2])
      list_lp2.append(lp2)

   LP = []
   for i in range(len(list_lp1)):
      height,width,depth = list_lp1[i].shape
      lp = numpy.hstack((list_lp1[i][:,0:width/2,:],list_lp2[i][:,width/2:,:]))
      LP.append(lp)

   lp = LP[0]
   for i in range(1,len(LP)):
      height,width = LP[i].shape[0:2]
      lp = cv2.pyrUp(lp,dstsize=(width,height))
      lp = cv2.add(lp,LP[i])
   img_in1 = lp
   img_out = img_in1 # Blending result

   return True, img_out

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);

   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)

   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   cv2.imwrite(output_name, output_image)

   return True

if __name__ == '__main__':
   question_number = -1

   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])

      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
