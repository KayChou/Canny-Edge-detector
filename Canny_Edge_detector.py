import numpy as np
import cv2
from scipy import ndimage, misc
from matplotlib import pyplot as plt
import math

def img_show(img):
	shape = img.shape
	'''
	dim1 = 600
	dim2 = int(dim1*shape[0]/shape[1])
	img = cv2.resize(img, (dim1, dim2))
	'''
	cv2.imshow("Image", img)
	# vmerge = np.vstack((im1, im2))
	cv2.waitKey()
	cv2.destroyAllWindows()


# ====================================================
# Gaussian filter
# ====================================================
def gs_filter(img):
	print("Start Gaussian Filter...")
	kernel = np.array(
		[[1, 4, 7, 4, 1],
		 [4, 16, 26, 16, 4],
		 [7, 26, 41, 26, 7],
		 [4, 16, 26, 16, 4],
		 [1, 4, 7, 4, 1]])/273
	img_gs = ndimage.filters.convolve(img, kernel)
	return img_gs


# ====================================================
# First order difference
# return the Amplitude and direction
# ====================================================
def gradient(img):
	kernel_x = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]])
	kernel_y = np.array(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]])

	gradient_x = ndimage.filters.convolve(img, kernel_x)
	gradient_y = ndimage.filters.convolve(img, kernel_y)

	amplitude = np.hypot(gradient_x, gradient_y)
	angle = np.arctan2(gradient_y, gradient_x)
	angle = (angle*180/math.pi) % 180
	# print("Angle Range: ", np.min(angle), np.max(angle))
	# angle must be in [0, 180]
	return amplitude, angle


# =======================================================
# Non-maximum suppression
# =======================================================
def suppression(img, angle):
	print("Start Non-maximum suppression...")
	max_grd = np.zeros_like(img)
	M, N = img.shape

	# Input angle range: 0 ~ 180
	# digital image has 4 angles: 0, 45, 90, 135
	for i in range(M):
		for j in range(N):
			if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] < 180):
				angle[i, j] = 0
			elif (22.5 <= angle[i, j] < 67.5):
				angle[i, j] = 45
			elif (67.5 <= angle[i, j] < 112.5):
				angle[i, j] = 90
			elif (112.5 <= angle[i, j] < 157.5):
				angle[i, j] = 135

	for i in range(1, M-1):
		for j in range(1, N-1):
			ang = angle[i, j]
			if ang == 0:
			    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
			        max_grd[i,j] = img[i,j]
			elif ang == 90:
			    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
			        max_grd[i,j] = img[i,j]
			elif ang == 135:
			    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
			        max_grd[i,j] = img[i,j]
			elif ang == 45:
			    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
			        max_grd[i,j] = img[i,j]

	return max_grd


def threshold(img, low, high):
	print("Double Thresholding...")
	img[np.where(img>high)] = 255
	img[np.where(img<low)] = 0
	img[np.where((img >= low) & (img<=high))] = 50
	return img


def main():
	# img = cv2.imread("lena.jpg")
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	print("Reading Source Image...")
	img = misc.imread("lena.jpg", flatten=True)
	print ("\tSource Image: ", img.shape)

	img1 = gs_filter(img)
	amp, angle = gradient(img1)
	img2 = suppression(amp, angle)
	img3 = threshold(img2, 40, 60)
	img_show(img3)
	print("Done!")


if __name__ == '__main__':
	main()


'''
def img_show(img1, img2):
	shape = img1.shape
	dim1 = 600
	dim2 = int(dim1*shape[0]/shape[1])

	im1 = cv2.resize(img1, (dim1, dim2))
	im2 = cv2.resize(img2, (dim1, dim2))
	hmerge = np.hstack((im1, im2))
	cv2.imshow("Image", hmerge)
	# vmerge = np.vstack((im1, im2))
	cv2.waitKey()
	cv2.destroyAllWindows()

img = cv2.imread("lena.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
'''