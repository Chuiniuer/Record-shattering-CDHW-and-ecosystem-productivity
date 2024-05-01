# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:59:39 2024

@author: Bohao Li
"""

from osgeo import gdal
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import os

def write_image(out_path, array, xsize, ysize, gt, proj):
    #writing new raster data
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    out_ds = driver.Create(out_path, xsize=xsize, ysize=ysize, bands=1, eType=gdal.GDT_Float32, options=["COMPRESS=LZW"])
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(np.nan)
    out_band.FlushCache()

    out_band = None
    out_ds = None

def main(l):
    input_folder= "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_Tmax_daily/"
    output_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/threshold/"
    
    ds = gdal.Open(r"/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/warm_seasons.tif")
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    #分块数
    blockx = 20
    blocky = 20
    #分块后每块的大小（除最后一块）
    bxsize = int(xsize / blockx) + 1
    bysize = int(ysize / blocky) + 1
    

    
    for c in range(blockx):
        if f"90th_{c}_{l}.tif" in os.listdir(output_folder):
            continue
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
        ws_arr = gdal.Open(r"/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/warm_seasons.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                          ysize=out_ysize)
        out_arr_90th = np.zeros(ws_arr.shape, dtype=np.float32)
        out_arr_25th = np.zeros(ws_arr.shape, dtype=np.float32)
        out_arr_75th = np.zeros(ws_arr.shape, dtype=np.float32)
        arr_list = []
        count = 0
        for m in range(1, 13):

            for y in range(1951, 1991):
                if y == 1951:
                    temp_arr = gdal.Open(input_folder + f"ERA5_T2mMax_Daily_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                      ysize=out_ysize)
                else:
                    temp_arr = np.concatenate((temp_arr, gdal.Open(input_folder + f"ERA5_T2mMax_Daily_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                      ysize=out_ysize)), axis=0)
            arr_list.append(temp_arr)

        for i in range(ws_arr.shape[0]):
            for j in range(ws_arr.shape[1]):
                start_month = ws_arr[i, j]
                if ws_arr[i, j] == 255:
                    out_arr_90th[i, j] = np.nan
                    out_arr_25th[i, j] = np.nan
                    out_arr_75th[i, j] = np.nan
                    continue
                for m in range(5):
                    if m+start_month > 12:
                        m_index = m+start_month-13
    
                    else:
                        m_index = m+start_month-1
                    if m==0:

                        time_series = arr_list[m_index][:, i, j]
                    else:

                        time_series = np.concatenate((time_series, arr_list[m_index][:, i, j]), axis=0)
                time_series = time_series.astype(np.float32)
                time_series[time_series==-32768] = np.nan
                time_series = time_series - 273.15
                print(time_series.shape)
                out_arr_90th[i, j] = np.percentile(time_series, 90)
                out_arr_25th[i, j] = np.percentile(time_series, 25)
                out_arr_75th[i, j] = np.percentile(time_series, 75)
        new_gt = (gt[0]+gt[1]*c*bxsize, gt[1], 0, gt[3]+gt[5]*l*bysize, 0, gt[5])
        write_image(output_folder+f"90th_{c}_{l}.tif", out_arr_90th, out_xsize,
                                   out_ysize, new_gt, proj)
        write_image(output_folder+f"25th_{c}_{l}.tif", out_arr_25th, out_xsize,
                                   out_ysize, new_gt, proj)
        write_image(output_folder+f"75th_{c}_{l}.tif", out_arr_75th, out_xsize,
                                   out_ysize, new_gt, proj)
            
if __name__ == "__main__":
    blocky = 20
    blockys = list(range(blocky))
    
    for l in blockys:
        main(l)

    with ProcessPoolExecutor(max_workers=20) as pool:
        pool.map(main, blockys)

                
            
            
            
            
            
            
            