# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 23:06:43 2024

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
    
    
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)            
		print("---  new folder...  ---")
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---")

def mission(y):
    if y < 2015:
        scenarios = ["historical"]
    else:
        scenarios = ["ssp126", "ssp245", "ssp585"]
    for scenario in scenarios:
        # iterate the models
        models = ["ACCESS-ESM1-5", "CanESM5", "CMCC-ESM2", "EC-Earth3-Veg",
                  "EC-Earth3-Veg-LR", "IPSL-CM6A-LR", "MPI-ESM1-2-LR", "NorESM2-LM", "NorESM2-MM"]
        
        for model in models:
            input_folder = f"F:/CMIP6_eco/heatwaves/HW_seve_monthly/{model}/{scenario}/"
            hw_severity_folder = f"F:/CMIP6_eco/heatwaves/HW_seve_annual/{model}/{scenario}/"
            mkdir(hw_severity_folder)
            ds = gdal.Open(input_folder + f"hw_seve_{model}_{scenario}_{y}.tif")
            ref_arr = ds.ReadAsArray()
            gt = ds.GetGeoTransform()
            hw = ds.ReadAsArray()
            hw_seve = hw.sum(axis=0)
            write_image(f"HW_severity_{model}_{scenario}", hw_severity_folder, y, hw_seve, hw_seve.shape[1], hw_seve.shape[0], gt)


if __name__ == "__main__":
    
    years = range(1950, 2100)
    for y in years:
        mission(y)
    
    
    