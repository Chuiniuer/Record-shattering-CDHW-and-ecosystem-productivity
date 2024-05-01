# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:59:06 2024

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
    
def hw_seve_cal(heat_begin, tmax, th25, th75, tmax_NTY=None):
    """
    Parameters
    ----------
    heat_begin : numpy.ndarray, 1-dimension
        the time series indicate heatwave begin node and duiration.
    tmax : numpy.ndarray, 1-dimension
        daily temperature maximums.
    tmax_NTY : numpy.ndarray, 1-dimension
        next year daily temperature maximums.
    th25 : float
        the 25 percentile maximum temperatures during 1951-1990.
    th75 : float
        DESCRIPTION.

    Returns
    -------
    heatwave severities in this year, heatwave severities in the next year.

    """
    result = np.zeros(tmax.shape)
    if tmax_NTY is not None:
        n_y_result = np.zeros(tmax_NTY.shape)
    for i in range(heat_begin.shape[0]):
        if heat_begin[i] == 255 and (tmax_NTY is not None):
            return np.full(tmax.shape, np.nan), np.full(tmax_NTY.shape, np.nan)
        elif heat_begin[i] == 255 and (tmax_NTY is None):
            return np.full(tmax.shape, np.nan)
        else:
            if heat_begin[i] == 0:
                continue
            else:
                for j in range(heat_begin[i]):
                    if i + j < heat_begin.shape[0]:
                        #the iteration of days during heatwaves
                        result[i+j] = (tmax[i+j] - th25) / (th75 - th25)
                    else:
                        it = i+j-heat_begin.shape[0]
                        n_y_result[it] = (tmax_NTY[it] - th25) / (th75 - th25)
    if tmax_NTY is not None:
        return result, n_y_result
    else:
        return result
                
    
def main(l):
    hw_bg_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/heatwave_begin_result/"
    tasmax_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_Tmax_daily/"

    output_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/hw_severity/"
    
    ds = gdal.Open(r"/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/th_merged/25th_era5.tif")
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    # number of blocks
    blockx = 20
    blocky = 40
    # sizes of blocks excepting the last one
    bxsize = int(xsize / blockx) + 1
    bysize = int(ysize / blocky) + 1
    

    
    for c in range(blockx):
        # if f"90th_{c}_{l}.tif" in os.listdir(output_folder):
        #     continue
        if l!=(blocky-1) and c!=(blockx-1):
            out_xsize = bxsize
            out_ysize = bysize
        elif l!=(blocky-1) and c==(blockx-1):
            out_xsize = xsize - (blockx-1) * bxsize
            out_ysize = bysize
        elif l==(blocky-1) and c!=(blockx-1):
            out_xsize = bxsize
            out_ysize = ysize - (blocky-1) * bysize
        else:
            out_xsize = xsize - (blockx-1) * bxsize
            out_ysize = ysize - (blocky-1) * bysize
        new_gt = (gt[0]+gt[1]*c*bxsize, gt[1], 0, gt[3]+gt[5]*l*bysize, 0, gt[5])
        
        th25_arr = gdal.Open(r"/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/th_merged/25th_era5.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                          ysize=out_ysize)
        th75_arr = gdal.Open(r"/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/th_merged/75th_era5.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                          ysize=out_ysize)
        for y in range(1951, 2023):
            hw_bg_path = hw_bg_folder + f"HeatWave_{y}.tif"
    
            if y != 2022:
                hw_bg = gdal.Open(hw_bg_path).ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                  ysize=out_ysize)
                
                for m in range(1, 13):
                    if m == 1:
                        tmax = gdal.Open(tasmax_folder + f"ERA5_T2mMax_Daily_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                          ysize=out_ysize)
                        tmax_nty = gdal.Open(tasmax_folder + f"ERA5_T2mMax_Daily_{y+1}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                          ysize=out_ysize)
                    else:
                        tmax = np.concatenate((tmax, gdal.Open(tasmax_folder + f"ERA5_T2mMax_Daily_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                          ysize=out_ysize)), axis=0)
                        tmax_nty = np.concatenate((tmax_nty, gdal.Open(tasmax_folder + f"ERA5_T2mMax_Daily_{y+1}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                          ysize=out_ysize)), axis=0)
                tmax[tmax==-32768] = np.nan
                tmax_nty[tmax_nty==-32768] = np.nan
                tmax = tmax - 273.15
                tmax_nty = tmax_nty - 273.15
                hw_seve = np.zeros(tmax.shape)
                hw_seve_nty = np.zeros(tmax_nty.shape)
                for i in tqdm(range(hw_bg.shape[1])):
                    for j in range(hw_bg.shape[2]):
                        hw_seve[:, i, j], hw_seve_nty[:, i, j] = hw_seve_cal(hw_bg[:, i, j], tmax[:, i, j], th25_arr[i, j], th75_arr[i, j], tmax_NTY=tmax_nty[:, i, j])
                write_3d_tif(f"hw_seve_{c}_{l}", output_folder, y, hw_seve, new_gt)
                write_3d_tif(f"hw_seve_nty_start_{c}_{l}", output_folder, y, hw_seve_nty, new_gt)
            else:
                hw_bg = gdal.Open(hw_bg_path).ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                  ysize=out_ysize)
                for m in range(1, 13):
                    if m == 1:
                        tmax = gdal.Open(tasmax_folder + f"ERA5_T2mMax_Daily_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                          ysize=out_ysize)
                    else:
                        tmax = np.concatenate((tmax, gdal.Open(tasmax_folder + f"ERA5_T2mMax_Daily_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                          ysize=out_ysize)), axis=0)
                        
                tmax[tmax==-32768] = np.nan
                tmax = tmax - 273.15
                hw_seve = np.zeros(tmax.shape)
                for i in tqdm(range(hw_bg.shape[1])):
                    for j in range(hw_bg.shape[2]):
                        hw_seve[:, i, j] = hw_seve_cal(hw_bg[:, i, j], tmax[:, i, j], th25_arr[i, j], th75_arr[i, j])
                write_3d_tif(f"hw_seve_{c}_{l}", output_folder, y, hw_seve, new_gt)
        

if __name__ == "__main__":
   blocky = 40
   blockys = list(range(blocky))
   
   for l in blockys:
       main(l)

   with ProcessPoolExecutor(max_workers=40) as pool:
       pool.map(main, blockys)









