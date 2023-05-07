#%%
# Imports

from skimage.io import imread
from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

#%%

"""
Settings
"""

__Fpath__="../../"

img1 = "Data/Lineal2/Future.jpg"

#%%
"""
Singular Value Decomposition
"""
# define a matrix
X = np.array([[3, 3, 2], [2, 3, -2]])
print(X)
# perform SVD
U, singular, V_transpose = svd(X)
# print different components
print("U: ", U)
print("Singular array", singular)
print("V^{T}", V_transpose)
#%%
"""
Calculate Pseudo inverse
"""
# inverse of singular matrix is just the reciprocal of each element
singular_inv = 1.0 / singular
# create m x n matrix of zeroes and put singular values in it
s_inv = np.zeros(X.shape)
s_inv[0][0] = singular_inv[0]
s_inv[1][1] = singular_inv[1]
# calculate pseudoinverse
M = np.dot(np.dot(V_transpose.T, s_inv.T), U.T)
print(M)
#%%
"""
SVD on image compression
"""

img = imread(__Fpath__+img1)
plt.imshow(img)
# convert to grayscale
gray_img = rgb2gray(img)

# calculate the SVD and plot the image
U, S, V_T = svd(gray_img, full_matrices=False)
S = np.diag(S)
fig, ax = plt.subplots(5, 2, figsize=(8, 20))
 
curr_fig = 0
for r in [5, 10, 70, 100, 200]:
    img_approx = U[:, :r] @ S[0:r, :r] @ V_T[:r, :]
    ax[curr_fig][0].imshow(img_approx, cmap='gray')
    ax[curr_fig][0].set_title("k = "+str(r))
    ax[curr_fig, 0].axis('off')
    ax[curr_fig][1].set_title("Original Image")
    ax[curr_fig][1].imshow(gray_img, cmap='gray')
    ax[curr_fig, 1].axis('off')
    curr_fig += 1
plt.show()
# %%
