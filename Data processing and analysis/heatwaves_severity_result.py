# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:57:23 2024

@author: Bohao Li
"""

from osgeo import gdal, osr
import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    input_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\heatwaves\\hw_seve_merged\\"
    out_folder = "E:\\l3\\创纪录极端复合干旱热浪事件影响生态系统生产力\\output\\heatwaves\\hw_seveity_merged_result\\"
    for y in range(1952, 2023):
        ds = gdal.Open(input_folder + f"hw_seve_{y}.tif")
        hw_seve = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        hw_seve_tsy_start = gdal.Open(input_folder + f"hw_seve_nty_start_{y-1}.tif").ReadAsArray()
        hw_seve[hw_seve_tsy_start!=0] = hw_seve_tsy_start[hw_seve_tsy_start!=0]
        write_3d_tif("hw_seve", out_folder, y, hw_seve, gt)
        