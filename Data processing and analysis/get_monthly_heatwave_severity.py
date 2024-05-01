# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:32:38 2024

@author: Bohao Li
"""
import pandas as pd
import numpy as np
from osgeo import gdal, osr
from datetime import datetime, timedelta

def write_3d_tif(variable_name, output_folder, array, geotransform):
    bands = array.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = output_folder + variable_name + '.tif'
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


if __name__ == "__main__":
    
    input_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\heatwaves\\hw_seveity_merged_result\\"
    out_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\heatwaves\\HW_seve_month\\"
    
    for y in range(1951, 2023):
        ds = gdal.Open(input_folder + f"hw_seve_{y}.tif")
        arr = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        start_index = 0
        temp_result = np.full((12, arr.shape[1], arr.shape[2]), np.nan)
        for m in range(1, 13, 1):
            start_date = datetime(y, m, 1)
            if m+1>12:
                temp_m = 0
                temp_y = y+1
                end_time = datetime(temp_y, temp_m+1, 1)+timedelta(days=-1)
            else:
                end_time = datetime(y, m+1, 1)+timedelta(days=-1)
            days = len(pd.date_range(start_date, end_time).strftime("%Y-%m-%d"))
            result = arr[start_index: start_index+days].sum(axis=0)
            temp_result[m-1] = result
            start_index += days
        write_3d_tif(f"hw_seve_{y}", out_folder, temp_result, gt)
    
            
        