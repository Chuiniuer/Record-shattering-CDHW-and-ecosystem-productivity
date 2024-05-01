# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:42:42 2024

@author: Bohao Li
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:00:24 2024

@author: Bohao Li
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
from osgeo import gdal, osr
import os
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import xarray as xr
import time
from concurrent.futures import ProcessPoolExecutor

def write_3d_tif(variable_name, output_folder, year, array, geotransform):
    bands = array.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = output_folder + variable_name + '_'  + str(year) + '.tif'
    out_tif = driver.Create(out_tif_name, array.shape[2], array.shape[1],
                            bands, gdal.GDT_Float32, options=["COMPRESS=LZW"])

    # Setting the image display range
    out_tif.SetGeoTransform(geotransform)

    # Get geographic coordinate system
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # Define the output coordinate system as WGS84
    out_tif.SetProjection(srs.ExportToWkt())  # Creating projection information for new layers

    for b in range(bands):

        # writing the results
        out_tif.GetRasterBand(b+1).WriteArray(array[b])
        out_tif.GetRasterBand(b+1).SetNoDataValue(np.nan)
    out_tif.FlushCache()
    # print(f'output successfully')
    del out_tif

def write_image(variable_name, output_folder, year, array, xsize, ysize, gt):
    out_tif_name = output_folder + variable_name + '_'  + str(year) + '.tif'
    #writing new raster data
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    out_ds = driver.Create(out_tif_name, xsize=xsize, ysize=ysize, bands=1, eType=gdal.GDT_Float32, options=["COMPRESS=LZW"])
    out_ds.SetGeoTransform(gt)
    # Get geographic coordinate system
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # Define the output coordinate system as WGS84
    out_ds.SetProjection(srs.ExportToWkt())  # Creating projection information for new layers
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(np.nan)
    out_band.FlushCache()

    out_band = None
    out_ds = None
    
if __name__ == "__main__":
    input_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\CDHW_seve\\CDHW_severity\\"
    CDHW_severity_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\CDHW_seve\CDHW_severity_annual\\"
    for y in range(1952, 2023):
        ds = gdal.Open(input_folder + f"CDHW_seve_{y}.tif")
        ref_arr = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        cdhw = ds.ReadAsArray()
        cdhw_seve = cdhw.sum(axis=0)
        write_image("CDHW_severity", CDHW_severity_folder, y, cdhw_seve, cdhw_seve.shape[1], cdhw_seve.shape[0], gt)
    
    
    
    
    
    
    
    