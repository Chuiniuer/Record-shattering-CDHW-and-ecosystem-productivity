# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 19:44:45 2024

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
    
    hw_seve_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\heatwaves\\HW_seve_month\\"
    dt_seve_folder = "G:\\SPEI_calculation\\from_era5\\drought_events\\"
    output_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\CDHW_seve\\CDHW_severity\\"
    
    for y in range(1952, 2023):
        ref_ds = gdal.Open(dt_seve_folder + f"DT_events_{y}.tif")
        gt = ref_ds.GetGeoTransform()
        dt_arr = ref_ds.ReadAsArray()
        dt_arr[np.isnan(dt_arr)] = 0
        
        hw_arr = gdal.Open(hw_seve_folder + f"hw_seve_{y}.tif").ReadAsArray()
        
        cdhw_seve = hw_arr * dt_arr
        
        write_3d_tif(f"CDHW_seve_{y}", output_folder, cdhw_seve, gt)