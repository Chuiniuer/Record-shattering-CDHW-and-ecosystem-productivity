# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:41:38 2024

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
                            bands, gdal.GDT_Byte, options=["COMPRESS=LZW"])

    # Setting the image display range
    out_tif.SetGeoTransform(geotransform)

    # Get geographic coordinate system
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # Define the output coordinate system as WGS84
    out_tif.SetProjection(srs.ExportToWkt())  # Creating projection information for new layers

    for b in range(bands):

        # writing the results
        out_tif.GetRasterBand(b+1).WriteArray(array[b])
        out_tif.GetRasterBand(b+1).SetNoDataValue(255)
    out_tif.FlushCache()
    # print(f'output successfully')
    del out_tif
    
    

if __name__ == "__main__":
    
    input_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\heatwaves\\heatwave_begin\\"
    output_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\heatwaves\\heatwave_begin_result\\"
    
    gt = gdal.Open(input_folder+"HeatWave_1951.tif").GetGeoTransform()
    for y in range(1951, 2023):
        if y == 1951:
            result = gdal.Open(input_folder + f"HeatWave_{y}.tif").ReadAsArray()
            write_3d_tif("HeatWave", output_folder, y, result, gt)
        else:
            hw = gdal.Open(input_folder + f"HeatWave_{y}.tif").ReadAsArray()
            is_begin = gdal.Open(input_folder + f"NextYearBeginHeat_{y}.tif").ReadAsArray()
            result = hw.copy()
            result[0][is_begin==1] = 0
            # result[result==-1] = 0
            write_3d_tif("HeatWave", output_folder, y, result, gt)

