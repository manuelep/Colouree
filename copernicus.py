# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:21:56 2019

@author: Colouree
"""

import geopandas as gpd
import fiona
import geopandas as gpd
import cartopy as ctpy
import matplotlib.pyplot as plt
gdf = gpd.read_file(r'C:\Users\Colouree\Desktop\Colouree\Copernicus Data\clc2018_cha1218_v2018_20_geoPackage\CLC2018_CHA1218_V2018_20.gpkg')#, layer='111: Continuous urban fabric')
geom=gdf['geometry']
from descartes import PolygonPatch

#def plot_gpd(stations):
#    # Reproj
#    crs = ctpy.crs.Mollweide()
#    crs_proj4 = crs.proj4_init
#    geom_reproj = stations.geometry.to_crs(crs_proj4)
#    # Plot
#    fig = plt.figure()
#    ax = fig.add_subplot(1, 1, 1, projection=crs)
#    ax.add_feature(ctpy.feature.COASTLINE)
#    geom_reproj.plot(ax=ax, markersize=3)
#    plt.savefig('plot_gpd.png')     

#def plot_ctpy(stations):
#    # Reproj
#    crs = ctpy.crs.Mollweide()
#    crs_proj4 = crs.proj4_init
#    geom_reproj = stations.geometry.to_crs(crs_proj4)
#    # Plot
#    fig = plt.figure()
#    ax = fig.add_subplot(1, 1, 1, projection=crs)
#    ax.add_feature(ctpy.feature.COASTLINE)
#    ax.add_geometries(geom_reproj, crs=crs)
#    plt.savefig('plot_ctpy.png')

#from PIL import Image
#Image.MAX_IMAGE_PIXELS = None
#import numpy as np
##read image
##img = Image.open("WWPI_2015_020m_eu_03035_d06_E40N20.TIF")
#image_tiff = Image.open(r'C:\Users\Colouree\Desktop\Colouree\Copernicus Data\clc2018_cha1218_v2018_20_raster100m\CLC2018_CHA1218_V2018_20.tif')

#stations = gpd.read_file(map_data)

#m_polygon = gdf['geometry']
#poly=[]
#if m_polygon.geometry == 'MULTIPOLYGON':
#    for pol in m_polygon:
#        poly.append(PolygonPatch(pol))
#else:
#    poly.append(PolygonPatch(m_polygon))
##df_map_elements.set_value(self_index, 'mpl_polygon', poly)
#import geopands as gpd
#from shapely.geometry.polygon import Polygon
#from shapely.geometry.multipolygon import MultiPolygon
#
#def explode(indata):
#    indf = gpd.GeoDataFrame.from_file(indata)
#    outdf = gpd.GeoDataFrame(columns=indf.columns)
#    for idx, row in indf.iterrows():
#        if type(row.geometry) == Polygon:
#            outdf = outdf.append(row,ignore_index=True)
#        if type(row.geometry) == MultiPolygon:
#            multdf = gpd.GeoDataFrame(columns=indf.columns)
#            recs = len(row.geometry)
#            multdf = multdf.append([row]*recs,ignore_index=True)
#            for geom in range(recs):
#                multdf.loc[geom,'geometry'] = row.geometry[geom]
#            outdf = outdf.append(multdf,ignore_index=True)
#    return outdf

#fig, ax = plt.subplots()
#for c_l ,patches in dict_mapindex_mpl_polygon.items():
#    p = PatchCollection(patches,color='white',lw=.3,edgecolor='k')
#    ax.add_collection(p)
#ax.autoscale_view()
#
#plt.show()

#
#import matplotlib.pyplot as plt 
#from descartes import PolygonPatch
#BLUE = '#6699cc'
#poly= test['geometry'][2]
#fig = plt.figure() 
#ax = fig.gca() 
#ax.add_patch(PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2 ))
#ax.axis('scaled')
#plt.show()
#


