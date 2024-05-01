# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:04:49 2024

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
import time

def get_warm_season_start(merged_arr):
    """

    Parameters
    ----------
    merged_arr : numpy.ndarray
        monthly time series.

    Returns
    -------
    numpy.ndarray.
    
    start month of warm seasons

    """
    monthly_mean = np.zeros((12, merged_arr.shape[1], merged_arr.shape[2]))
    years = int(merged_arr.shape[0] / 12)
    temp_arr = np.zeros((years, merged_arr.shape[1], merged_arr.shape[2]))
    for i in range(12):
        # monthly iteration
        for j in range(years):
            temp_arr[j] = merged_arr[i+j*12]
        
        # save as monthly mean temp array
        monthly_mean[i] = temp_arr.mean(axis=0)
    
    result = np.zeros((12, merged_arr.shape[1], merged_arr.shape[2]))
    # getting the start month of warm season
    for i in range(12):
        for j in range(5):
            if i+j > 11:
                result[i] += monthly_mean[i+j-12]
            else:
                result[i] += monthly_mean[i+j]
    id_result = np.argmax(result, axis=0)
    id_result = id_result + 1
    return id_result

def write_tif(out_tif_name, gt, proj, nodata, array):
    driver = gdal.GetDriverByName('GTiff')
    out_tif = driver.Create(out_tif_name, array.shape[1], array.shape[0], 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])

    out_tif.SetGeoTransform(gt)

    out_tif.SetProjection(proj)  # Creating projection information for new layers

    # writing the results
    out_tif.GetRasterBand(1).WriteArray(array)
    out_tif.GetRasterBand(1).SetNoDataValue(nodata)
    out_tif.FlushCache()
    # print(f'output successfully')
    del out_tif

if __name__ == "__main__":
    
    input_folder= "G:\\ERA5-land\\ERA5_tas_Monthly\\"
    output_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\"
    
    ref_ds = gdal.Open(input_folder+"ERA5_tas_Monthly_1951_1.tif")
    ref_arr = ref_ds.ReadAsArray()
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    
    temp = np.zeros((72*12, ref_arr.shape[0], ref_arr.shape[1]), dtype=np.float32)
    count = 0
    for y in range(1951, 2023):
        for m in range(1, 13):
            temp[count] = gdal.Open(input_folder+f"ERA5_tas_Monthly_{y}_{m}.tif").ReadAsArray()
            count += 1
    temp[temp==-32768] = np.nan
    result = get_warm_season_start(temp)
    result[ref_arr==-32768] = 255
    
    write_tif(output_folder+"warm_seasons.tif", gt, proj, 255, result)
    
    
    
    
    