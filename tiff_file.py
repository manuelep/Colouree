# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:46:44 2019

@author: Colouree
"""

#from PIL import Image
#image_tiff = Image.open('s1a-iw1-slc-vh-20190912t055158-20190912t055224-028984-034991-001.tiff')
#image_tiff.show()
#
import os, gdal

in_path = 'C:/Users/Colouree/Desktop/Colouree/'
input_filename = 's1a-iw1-slc-vh-20190912t055158-20190912t055224-028984-034991-001.tif'

out_path = 'C:/Users/Colouree/Desktop/Colouree/'
output_filename = 'tile_'

tile_size_x = 50
tile_size_y = 70

ds = gdal.Open(in_path + input_filename)
band = ds.GetRasterBand(1)
#xsize = band.XSize
#ysize = band.YSize
#
#for i in range(0, xsize, tile_size_x):
#    for j in range(0, ysize, tile_size_y):
#        com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
#        os.system(com_string)

#
#image_path='s1a-iw1-slc-vh-20190912t055158-20190912t055224-028984-034991-001.tiff'
#from PIL import TiffImagePlugin
#TiffImagePlugin.DEBUG = True
#with open(image_path, 'rb') as f:
#    TiffImagePlugin.TiffImageFile(f)
        
        ######################################################################
import numpy
        
arr = band.ReadAsArray()
[cols, rows] = arr.shape
arr_min = arr.Min()
arr_max = arr.Max()
arr_mean = int(arr.mean())
arr_out = numpy.where((arr < arr_mean), 10000, arr)
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(output_filename, rows, cols, 1, gdal.GDT_UInt16)
outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(ds.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(arr_out)
outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!
outdata = None
band=None
ds=None