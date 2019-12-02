from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
#read image
#img = Image.open("WWPI_2015_020m_eu_03035_d06_E40N20.TIF")
image_tiff = Image.open('s1a-iw1-slc-vh-20190912t055158-20190912t055224-028984-034991-001.tif')

#img = Image.open("imm.png")
# k=np.asarray(img)
# w,h=np.shape(k)

#view the image 
# img.show()
# ll=[]
#############################################################################
# from libtiff import TIFF

# tif = TIFF.open('WWPI_2015_020m_eu_03035_d06_E40N20.TIF') # open tiff file in read mode
# # read an image in the currect TIFF directory as a numpy array
# image = tif.read_image()

# # read all images in a TIFF file:
# for image in tif.iter_images(): 
#     pass

# tif = TIFF.open('WWPI_2015_020m_eu_03035_d06_E40N20.TIF', mode='w')
# tif.write_image(image)
##################################################################################
# from tqdm import tqdm
# for i in tqdm(range(0,w)):
#    for j in range(0,h):
#        if k[i][j]>0:
#            ll.append(k[i][j])
#3333333333333333333333333333333333333333333333333333333333333333333333333333
# import matplotlib.pyplot as plt
# #read image
# img_arr = plt.imread("200km_2p5m_N24E40.TIF")

# #view image
# plt.imshow(img_arr)
# plt.show()
##########################################################################

# import os
# import numpy as np
# # File manipulation
# from glob import glob
# import matplotlib.pyplot as plt
# import geopandas as gpd
# import rasterio as rio
# import earthpy as et
# import earthpy.spatial as es
# import earthpy.plot as ep

##########################################################################
# from osgeo import gdal
# import numpy as np

# ds = gdal.Open("WWPI_2015_020m_eu_03035_d06_E40N20.TIF")

# # loop through each band
# for bi in range(ds.RasterCount):
#     band = ds.GetRasterBand(bi + 1)
#     # Read this band into a 2D NumPy array
#     ar = band.ReadAsArray()
#     print('Band %d has type %s'%(bi + 1, ar.dtype))
#     raw = ar.tostring()



o = {
   "coordinates": [[[23.314208, 37.768469], [24.039306, 37.768469], [24.039306, 38.214372], [23.314208, 38.214372], [23.314208, 37.768469]]], 
   "type": "Polygon"
}
https://github.com/acgeospatial/Satellite_Imagery_Python/blob/master/bounds_IOW.geojson
