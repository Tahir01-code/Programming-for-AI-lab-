import cv2
from matplotlib import pyplot as plt

image_path = r"C:\Users\tahir\Desktop\task 5\tahir pic.jpg"
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (1900, 800))
resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
plt.imshow(resized_image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()

## Blurring Image

Gaussian = cv2.GaussianBlur(resized_image, (15, 15), 0)  
Gaussian_rgb = cv2.cvtColor(Gaussian, cv2.COLOR_BGR2RGB)  
plt.imshow(Gaussian_rgb)
plt.title('Gaussian Blurred Image')
plt.axis('off')
plt.show()



# Median Blurred

median = cv2.medianBlur(resized_image, 11)  
median_rgb = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)  

plt.imshow(median_rgb)
plt.title('Median Blurred Image')
plt.axis('off')
plt.show()



##rayScaling_Image


# Method 1: Using the cv2.cvtColor() function


import cv2

image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0)  
cv2.destroyAllWindows()


# Method 3 Weighted Method (Recommended)

import cv2

img_weighted = cv2.imread(image_path)
rows, cols = img_weighted.shape[:2]

for i in range(rows):
    for j in range(cols):
        gray = 0.2989 * img_weighted[i, j][2] + 0.5870 * img_weighted[i, j][1] + 0.1140 * img_weighted[i, j][0]
        img_weighted[i, j] = [gray, gray, gray]

cv2.imshow('Grayscale Image (Weighted)', img_weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Image Resizig

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
scale_factor_1 = 3.0  
scale_factor_2 = 1/3.0
height, width = image_rgb.shape[:2]
new_height = int(height * scale_factor_1)
new_width = int(width * scale_factor_1)

zoomed_image = cv2.resize(src =image_rgb, 
                          dsize=(new_width, new_height), 
                          interpolation=cv2.INTER_CUBIC)
                          
new_height1 = int(height * scale_factor_2)
new_width1 = int(width * scale_factor_2)
scaled_image = cv2.resize(src= image_rgb, 
                          dsize =(new_width1, new_height1), 
                          interpolation=cv2.INTER_AREA)

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].imshow(image_rgb)
axs[0].set_title('Original Image Shape:'+str(image_rgb.shape))
axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed Image Shape:'+str(zoomed_image.shape))
axs[2].imshow(scaled_image)
axs[2].set_title('Scaled Image Shape:'+str(scaled_image.shape))

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()



## Image Rotation


import cv2
import matplotlib.pyplot as plt
img = cv2.imread(image_path)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)
angle = 30
scale = 1
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (img.shape[1], img.shape[0]))

fig, axs = plt.subplots(1, 2, figsize=(7, 4))
axs[0].imshow(image_rgb)
axs[0].set_title('Original Image')
axs[1].imshow(rotated_image)
axs[1].set_title('Image Rotation')
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()



## Image Translation


import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread(image_path)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
width, height = image_rgb.shape[1], image_rgb.shape[0]

tx, ty = 100, 70
translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
translated_image = cv2.warpAffine(image_rgb, translation_matrix, (width, height))

fig, axs = plt.subplots(1, 2, figsize=(7, 4))
axs[0].imshow(image_rgb), axs[0].set_title('Original Image')
axs[1].imshow(translated_image), axs[1].set_title('Image Translation')

for ax in axs:
    ax.set_xticks([]), ax.set_yticks([])

plt.tight_layout()
plt.show()




## Image Shearing


import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
width, height = image_rgb.shape[1], image_rgb.shape[0]

shearX, shearY = -0.15, 0
transformation_matrix = np.array([[1, shearX, 0], [0, 1, shearY]], dtype=np.float32)
sheared_image = cv2.warpAffine(image_rgb, transformation_matrix, (width, height))

fig, axs = plt.subplots(1, 2, figsize=(7, 4))
axs[0].imshow(image_rgb), axs[0].set_title('Original Image')
axs[1].imshow(sheared_image), axs[1].set_title('Sheared Image')

for ax in axs:
    ax.set_xticks([]), ax.set_yticks([])

plt.tight_layout()
plt.show()



## Image Normalization

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
b, g, r = cv2.split(image_rgb)

b_normalized = cv2.normalize(b.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
g_normalized = cv2.normalize(g.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
r_normalized = cv2.normalize(r.astype('float'), None, 0, 1, cv2.NORM_MINMAX)

normalized_image = cv2.merge((b_normalized, g_normalized, r_normalized))
print(normalized_image[:, :, 0])

plt.imshow(normalized_image)
plt.xticks([]), 
plt.yticks([]), 
plt.title('Normalized Image')
plt.show()



## Edge Detection

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(image_path)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
edges = cv2.Canny(image_rgb, 100, 700)

fig, axs = plt.subplots(1, 2, figsize=(7, 4))
axs[0].imshow(image_rgb), axs[0].set_title('Original Image')
axs[1].imshow(edges), axs[1].set_title('Image Edges')

for ax in axs:
    ax.set_xticks([]), ax.set_yticks([])

plt.tight_layout()
plt.show()



## Morpholohical Image Processing

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)

dilated = cv2.dilate(image_gray, kernel, iterations=2)
eroded = cv2.erode(image_gray, kernel, iterations=2)
opening = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel)

fig, axs = plt.subplots(2, 2, figsize=(7, 7))
axs[0, 0].imshow(dilated, cmap='Greys'), axs[0, 0].set_title('Dilated Image')
axs[0, 1].imshow(eroded, cmap='Greys'), axs[0, 1].set_title('Eroded Image')
axs[1, 0].imshow(opening, cmap='Greys'), axs[1, 0].set_title('Opening')
axs[1, 1].imshow(closing, cmap='Greys'), axs[1, 1].set_title('Closing')

for ax in axs.flatten():
    ax.set_xticks([]), ax.set_yticks([])

plt.tight_layout()
plt.show()



## Intensity Transformation

## Log Transformation

import cv2
import numpy as np

# Open the image.
img = cv2.imread(image_path)

# Apply log transform.
c = 255/(np.log(1 + np.max(img)))
log_transformed = c * np.log(1 + img)

# Specify the data type.
log_transformed = np.array(log_transformed, dtype = np.uint8)

# Save the output.
cv2.imwrite('log_transformed.jpg', log_transformed)



## Power Law

import cv2
import numpy as np

# Open the image.
img = cv2.imread(image_path)

# Trying 4 gamma values.
for gamma in [0.1, 0.5, 1.2, 2.2]:
    
    # Apply gamma correction.
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8')

    # Save edited images.
    cv2.imwrite('gamma_transformed'+str(gamma)+'.jpg', gamma_corrected)



## Piecewise-Linear Transformation Functions

import cv2
import numpy as np

# Function to map each intensity level to output intensity level.
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2

# Open the image.
img = cv2.imread(image_path)

# Define parameters.
r1 = 70
s1 = 0
r2 = 140
s2 = 255

# Vectorize the function to apply it to each value in the Numpy array.
pixelVal_vec = np.vectorize(pixelVal)

# Apply contrast stretching.
contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)

# Save edited image.
cv2.imwrite('contrast_stretch.jpg', contrast_stretched)



## Image Translation

## Translating the Image Right and Down 

import cv2
import numpy as np

image = cv2.imread(
    image_path
)

# Check if image loaded
if image is None:
    print("Image not found!")
    exit()

height, width = image.shape[:2]
quarter_height, quarter_width = height / 4, width / 4

T = np.float32([[1, 0, quarter_width],
                [0, 1, quarter_height]])

img_translation = cv2.warpAffine(image, T, (width, height))

cv2.imshow("Original Image", image)
cv2.imshow("Translated Image", img_translation)

cv2.waitKey(0)
cv2.destroyAllWindows()




## Performing Multiple Translations


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(image_path)
rows, cols, _ = img.shape

M_left = np.float32([[1, 0, -50], [0, 1, 0]])
M_right = np.float32([[1, 0, 50], [0, 1, 0]])
M_top = np.float32([[1, 0, 0], [0, 1, 50]])
M_bottom = np.float32([[1, 0, 0], [0, 1, -50]])

img_left = cv2.warpAffine(img, M_left, (cols, rows))
img_right = cv2.warpAffine(img, M_right, (cols, rows))
img_top = cv2.warpAffine(img, M_top, (cols, rows))
img_bottom = cv2.warpAffine(img, M_bottom, (cols, rows))

plt.subplot(221), plt.imshow(img_left), plt.title('Left')
plt.subplot(222), plt.imshow(img_right), plt.title('Right')
plt.subplot(223), plt.imshow(img_top), plt.title('Top')
plt.subplot(224), plt.imshow(img_bottom), plt.title('Bottom')
plt.show()



## Image Pyramid

## Pyramid Down with cv2.pyrDown()

import cv2
import numpy as np

image = cv2.imread(image_path)

# Check if image loaded
if image is None:
    print("Image not found!")
    exit()

downsampled_image = cv2.pyrDown(image)

cv2.imshow("Original Image", image)
cv2.imshow("Downsampled Image (PyrDown)", downsampled_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


## 2. Pyramid Up with cv2.pyrUp()


import cv2
import numpy as np

image = cv2.imread(image_path)
# Check if image loaded
if image is None:
    print("Image not found!")
    exit()

upsampled_image = cv2.pyrUp(image)

cv2.imshow("Original Image", image)
cv2.imshow("Upsampled Image (PyrUp)", upsampled_image)

cv2.waitKey(0)
cv2.destroyAllWindows()



## Building a Gaussian Pyramid (Multiple Levels)


import cv2
import numpy as np

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

pyramid = [image]

for i in range(3):
    image = cv2.pyrDown(image)
    pyramid.append(image)

for i in range(len(pyramid)):
    cv2.imshow(f"Pyramid Level {i}", pyramid[i])

cv2.waitKey(0)
cv2.destroyAllWindows()


### Histograms Equalization

## Applying Histogram Equalization

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread(image_path, 0)  
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))
plt.figure(figsize=(10, 5))
plt.imshow(res, cmap='gray')  
plt.title("Original vs Equalized Image")
plt.axis('off')  
plt.show()



## Converting BGR to Grayscale

import cv2
src = cv2.imread(image_path)  # Read the image

# Convert to Grayscale
gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# Display
cv2.imshow("Grayscale Image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




## Convert BGR to HSV

import cv2
src = cv2.imread(image_path)

# Convert to HSV
hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

cv2.imshow("HSV Image", hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


## Convert BGR to RGB (For Matplotlib)

import cv2
import matplotlib.pyplot as plt
src = cv2.imread(image_path)

# Convert from BGR to RGB
rgb_image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

# Display with Matplotlib
plt.imshow(rgb_image)
plt.title("RGB Image for Matplotlib")
plt.axis('off')
plt.show()



## Visualizing image in different color spaces

## Python program to read image as RGB

#importing cv2 and matplotlib module
import cv2
import matplotlib.pyplot as plt

# reads image as RGB
img = cv2.imread(image_path)

# shows the image
plt.imshow(img)


# Grey Scale Image 
# 
# # Python program to read image as GrayScale

# Importing cv2 module
import cv2

# Reads image as gray scale
img = cv2.imread(image_path, 0) 

# We can alternatively convert
# image by using cv2color
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Shows the image
cv2.imshow('image', img) 

cv2.waitKey(0)         
cv2.destroyAllWindows() 



# ## Making Border

import cv2

image = cv2.imread(image_path)
image = cv2.copyMakeBorder( image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

cv2.imshow("Bordered Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



## different borders

import cv2

image = cv2.imread(image_path)

border_reflect = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_REFLECT)
border_reflect_101 = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_REFLECT_101)
border_replicate = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_REPLICATE)

cv2.imshow("Border Reflect", border_reflect)
cv2.imshow("Border Reflect 101", border_reflect_101)
cv2.imshow("Border Replicate", border_replicate)

cv2.waitKey(0)
cv2.destroyAllWindows()


## Red Boarders

import cv2

image = cv2.imread(image_path)
bordered_image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 255))

cv2.imshow("Red Border Image", bordered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

## -------------------------------------------------------------------------------------------------------

## Image Segmentation and Thresholding


import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

show_image(gray_image, 'Original Grayscale Image')

## Binary Thresholding

_, thresh_binary = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)
show_image(thresh_binary, 'Binary Threshold ')

## Binary Threshold Inverted

_, thresh_binary_inv = cv2.threshold(
    gray_image, 120, 255, cv2.THRESH_BINARY_INV)
show_image(thresh_binary_inv, 'Binary Threshold Inverted ')


## Truncated Threshold

_, thresh_trunc = cv2.threshold(gray_image, 120, 255, cv2.THRESH_TRUNC)
show_image(thresh_trunc, 'Truncated Threshold')

## To Zero Threshold

_, thresh_tozero = cv2.threshold(gray_image, 120, 255, cv2.THRESH_TOZERO)
show_image(thresh_tozero, 'Set to 0 ')

## To Zero Inverted Threshold

_, thresh_tozero_inv = cv2.threshold(
    gray_image, 120, 255, cv2.THRESH_TOZERO_INV)
show_image(thresh_tozero_inv, 'Set to 0 Inverted')


##--------------------------------------------------------------------------------------------------


### Adaptive Thresholding

import cv2
import matplotlib.pyplot as plt

image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# ## Adaptive Mean Thresholding

thresh_mean = cv2.adaptiveThreshold(
    gray_image, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    199, 5
)
show_image(thresh_mean, "Adaptive Mean Thresholding")


# ## Adaptive Gaussian Thresholding

thresh_gauss = cv2.adaptiveThreshold(
    gray_image, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    199, 5
)
show_image(thresh_gauss, "Adaptive Gaussian Thresholding")



##---------------------------------------------------------------------------------------------


## Otsu's Thresholding

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

ret, otsu_thresh = cv2.threshold(
    gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Calculated Otsu threshold value:", ret)
show_image(otsu_thresh, "Otsu’s Thresholding")





