#%%
# Imports

from skimage.io import imread
from scipy.linalg import svd 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns

#%%

"""
Settings
"""
sns.set_style('darkgrid')

plt.rcParams["font.family"] = "serif"

plt.rcParams["font.size"] = "16"

__Fpath__="../../"

img1 = "Data/Lineal2/Future.jpg"
img2 = "Data/Lineal2/Cardenal Pantanero.jpg"

#img3 = "Data/Lineal2/Montaña.jpg"

#%%
"""
Functions
"""

def channel_svd(channel):
    """
    method to calculate svd of the input data matrix
    :param channel: data matrix whose svd is to be calculatec
    :return: list of three matrices: U, Sigma and V transpose
    """
    [u, sigma, vt] = svd(channel)
    sigma = np.diag(sigma)
    return [u, sigma, vt]

def im2double(im):
    """method to get double precision of a channel"""
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max


def plotting(k,cR,cG,cB,ax,curr_fig,img):
    """
    method to plot the approximate image for the given k values
    k: k to be use.
    cR-G-B: SVD for a single color channel
    ax: the axis plot 
    curr_fig: the figure to be plot
    Return the plot for the given k
    """
    channels = [cR,cG,cB]
    channels_approx =[]
    for i in range(3):
        U = channels[i][0]
        S = channels[i][1]
        V_T = channels[i][2]
        channels_approx.append( U[:, :k] @ S[0:k, :k] @ V_T[:k, :])

    re_image = cv2.merge((channels_approx[2], channels_approx[1], channels_approx[0]))
    ax[curr_fig, 0].imshow(re_image)
    ax[curr_fig, 0].set_title("k = "+str(k))
    ax[curr_fig, 0].axis('off')
    ax[curr_fig, 1].set_title("Original Image")
    ax[curr_fig, 1].imshow(img)
    ax[curr_fig, 1].axis('off')


#%%
"""
Reading the image
"""
img = imread(__Fpath__+img2)
plt.imshow(img)
plt.axis("off")

"""
Aplaying compresion
"""
blue_channel = im2double(img[:, :, 0])
green_channel = im2double(img[:, :, 1])
red_channel = im2double(img[:, :, 2])

cR = channel_svd(red_channel)
cB = channel_svd(blue_channel)
cG = channel_svd(green_channel)


#%%
"""
Ploting img
"""

r = [5, 10, 70, 100, 200]

# calculate the SVD and plot the image

fig, ax = plt.subplots(5, 2, figsize=(8, 20))
 
curr_fig = 0
for k in r:
    plotting(k,cR,cG,cB,ax,curr_fig,img)
    curr_fig += 1
plt.show()

# %%
