# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 23:10:55 2024

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


def write_image(variable_name, output_folder, array, xsize, ysize, gt):
    out_tif_name = output_folder + variable_name + '.tif'
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
    input_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\heatwaves\\hw_severity_annual\\"
    std_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\heatwaves\\hw_std\\"
    max_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\heatwaves\\hw_max\\"
    composited_arr = np.array([gdal.Open(input_folder + f"HW_severity_{y}.tif").ReadAsArray()
     for y in range(1952, 1991)])
    gt = gdal.Open(input_folder+"HW_severity_1952.tif").GetGeoTransform()
    result = composited_arr.std(axis=0)
    write_image("HW_severity_std", std_folder, result, result.shape[1], result.shape[0], gt)
    
    
    composited_arr = np.array([gdal.Open(input_folder + f"HW_severity_{y}.tif").ReadAsArray()
     for y in range(1952, 1991)])
    thre = composited_arr.max(axis=0)
    write_image("HW_severity_max", max_folder, thre, result.shape[1], result.shape[0], gt)