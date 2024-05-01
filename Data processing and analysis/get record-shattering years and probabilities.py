# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:39:03 2024

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

def write_3d_tif(variable_name, output_folder, array, geotransform):
    bands = array.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = output_folder + variable_name + '.tif'
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
        out_tif.GetRasterBand(b+1).SetNoDataValue(2)
    out_tif.FlushCache()
    # print(f'output successfully')
    del out_tif


def record_shattering_years(seve_time_series, maximum, margin):
    
    if np.isnan(seve_time_series[0]):
        return np.full(seve_time_series.shape, 2)
    result = np.zeros(seve_time_series.shape)
    for i in range(seve_time_series.shape[0]):
        if seve_time_series[i] > maximum + margin:
            result[i] = 1
            maximum = seve_time_series[i]
        else:
            result[i] = 0
    return result

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   # Determine if a folder exists if it doesn't then create it as a folder
		os.makedirs(path)            #makedirs Creates a path if it does not exist when creating a file.
		print("---  new folder...  ---")
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---")
        
if __name__ == "__main__":
    
    scenarios = ["historical", "ssp126", "ssp245", "ssp585"]
    mask_ref_arr = gdal.Open(r"E:\l3\创纪录极端复合干旱热浪事件影响生态系统生产力\output\warm_seasons_resampled\warm_seasons.tif").ReadAsArray()
    mask_arr = (mask_ref_arr == 255)
    
    for scenario in scenarios:
        if scenario == "historical":
            years = range(1991, 2015)
        else:
            years = range(2015, 2100)
        models = ["ACCESS-ESM1-5", "CanESM5", "CMCC-ESM2", "EC-Earth3-Veg",
                  "EC-Earth3-Veg-LR", "IPSL-CM6A-LR", "MPI-ESM1-2-LR", "NorESM2-LM", "NorESM2-MM"]
        
        for model in models:
            input_folder = f"F:/CMIP6_eco/CDHW/CDHW_severity_annual/{model}/{scenario}/"
            rsr_ts_folder = f"F:/CMIP6_eco/CDHW/CDHW_record_shattering_time_series/{model}/{scenario}/"
            rsr_prb_folder = f"F:/CMIP6_eco/CDHW/CDHW_record_shattering_probabilities/{model}/{scenario}/"
            
            mkdir(rsr_prb_folder)
            mkdir(rsr_ts_folder)
            
            composited_arr = np.array([gdal.Open(input_folder + f"CDHW_severity_{model}_{scenario}_{y}.tif").ReadAsArray()
             for y in years])
            
            ds_max = gdal.Open(f"F:/CMIP6_eco/CDHW/maximum/{model}/CDHW_severity_{model}_historical_max.tif")
            gt = ds_max.GetGeoTransform()
            arr_max = ds_max.ReadAsArray()
            sigma = gdal.Open(f"F:/CMIP6_eco/CDHW/std/CDHW_severity_std.tif").ReadAsArray()
            
            ts_result_0sigma = np.zeros(composited_arr.shape, dtype=np.int8)
            ts_result_1sigma = np.zeros(composited_arr.shape, dtype=np.int8)
            ts_result_2sigma = np.zeros(composited_arr.shape, dtype=np.int8)
            ts_result_3sigma = np.zeros(composited_arr.shape, dtype=np.int8)
            prb_result = np.zeros(arr_max.shape)
            
            for i in tqdm(range(arr_max.shape[0])):
                for j in range(arr_max.shape[1]):
                    ts_result_0sigma[:, i, j] = record_shattering_years(composited_arr[:, i, j], arr_max[i, j], 0)
                    ts_result_1sigma[:, i, j] = record_shattering_years(composited_arr[:, i, j], arr_max[i, j], sigma[i, j])
                    ts_result_2sigma[:, i, j] = record_shattering_years(composited_arr[:, i, j], arr_max[i, j], 2*sigma[i, j])
                    ts_result_3sigma[:, i, j] = record_shattering_years(composited_arr[:, i, j], arr_max[i, j], 3*sigma[i, j])
            if scenario == "historical":
                write_3d_tif(f"CDHW_{model}_1991_2014_0sigma", rsr_ts_folder, ts_result_0sigma, gt)
                write_3d_tif(f"CDHW_{model}_1991_2014_1sigma", rsr_ts_folder, ts_result_1sigma, gt)
                write_3d_tif(f"CDHW_{model}_1991_2014_2sigma", rsr_ts_folder, ts_result_2sigma, gt)
                write_3d_tif(f"CDHW_{model}_1991_2014_3sigma", rsr_ts_folder, ts_result_3sigma, gt)
            
            
            
                rsp_arr_0sigma = ts_result_0sigma.sum(axis=0).astype(np.float32)/ ts_result_0sigma.shape[0]
                rsp_arr_1sigma = ts_result_1sigma.sum(axis=0).astype(np.float32)/ ts_result_1sigma.shape[0]
                rsp_arr_2sigma = ts_result_2sigma.sum(axis=0).astype(np.float32)/ ts_result_2sigma.shape[0]
                rsp_arr_3sigma = ts_result_3sigma.sum(axis=0).astype(np.float32)/ ts_result_3sigma.shape[0]
            
                rsp_arr_0sigma[ts_result_0sigma[0]==2] = np.nan
                rsp_arr_1sigma[ts_result_1sigma[0]==2] = np.nan
                rsp_arr_2sigma[ts_result_1sigma[0]==2] = np.nan
                rsp_arr_3sigma[ts_result_1sigma[0]==2] = np.nan
                
                rsp_arr_0sigma[mask_arr] = np.nan
                rsp_arr_1sigma[mask_arr] = np.nan
                rsp_arr_2sigma[mask_arr] = np.nan
                rsp_arr_3sigma[mask_arr] = np.nan
                
                
                write_image(f"CDHW_RSP_{model}_1991_2014_0sigma", rsr_prb_folder, rsp_arr_0sigma, rsp_arr_0sigma.shape[1], rsp_arr_0sigma.shape[0], gt)
                write_image(f"CDHW_RSP_{model}_1991_2014_1sigma", rsr_prb_folder, rsp_arr_1sigma, rsp_arr_1sigma.shape[1], rsp_arr_1sigma.shape[0], gt)
                write_image(f"CDHW_RSP_{model}_1991_2014_2sigma", rsr_prb_folder, rsp_arr_2sigma, rsp_arr_1sigma.shape[1], rsp_arr_1sigma.shape[0], gt)
                write_image(f"CDHW_RSP_{model}_1991_2014_3sigma", rsr_prb_folder, rsp_arr_3sigma, rsp_arr_1sigma.shape[1], rsp_arr_1sigma.shape[0], gt)
            
            else:
                
                write_3d_tif(f"CDHW_{model}_{scenario}_2015_2099_0sigma", rsr_ts_folder, ts_result_0sigma, gt)
                write_3d_tif(f"CDHW_{model}_{scenario}_2015_2099_1sigma", rsr_ts_folder, ts_result_1sigma, gt)
                write_3d_tif(f"CDHW_{model}_{scenario}_2015_2099_2sigma", rsr_ts_folder, ts_result_2sigma, gt)
                write_3d_tif(f"CDHW_{model}_{scenario}_2015_2099_3sigma", rsr_ts_folder, ts_result_3sigma, gt)
            
            
                # get mean 2020-2039
                rsp_arr_0sigma = ts_result_0sigma[5: 25].sum(axis=0).astype(np.float32)/ ts_result_0sigma[5: 25].shape[0]
                rsp_arr_1sigma = ts_result_1sigma[5: 25].sum(axis=0).astype(np.float32)/ ts_result_1sigma[5: 25].shape[0]
                rsp_arr_2sigma = ts_result_2sigma[5: 25].sum(axis=0).astype(np.float32)/ ts_result_2sigma[5: 25].shape[0]
                rsp_arr_3sigma = ts_result_3sigma[5: 25].sum(axis=0).astype(np.float32)/ ts_result_3sigma[5: 25].shape[0]
            
                rsp_arr_0sigma[ts_result_0sigma[0]==2] = np.nan
                rsp_arr_1sigma[ts_result_1sigma[0]==2] = np.nan
                rsp_arr_2sigma[ts_result_1sigma[0]==2] = np.nan
                rsp_arr_3sigma[ts_result_1sigma[0]==2] = np.nan
                
                rsp_arr_0sigma[mask_arr] = np.nan
                rsp_arr_1sigma[mask_arr] = np.nan
                rsp_arr_2sigma[mask_arr] = np.nan
                rsp_arr_3sigma[mask_arr] = np.nan
                
                write_image(f"CDHW_RSP_{model}_{scenario}_2020_2039_0sigma", rsr_prb_folder, rsp_arr_0sigma, rsp_arr_0sigma.shape[1], rsp_arr_0sigma.shape[0], gt)
                write_image(f"CDHW_RSP_{model}_{scenario}_2020_2039_1sigma", rsr_prb_folder, rsp_arr_1sigma, rsp_arr_1sigma.shape[1], rsp_arr_1sigma.shape[0], gt)
                write_image(f"CDHW_RSP_{model}_{scenario}_2020_2039_2sigma", rsr_prb_folder, rsp_arr_2sigma, rsp_arr_1sigma.shape[1], rsp_arr_1sigma.shape[0], gt)
                write_image(f"CDHW_RSP_{model}_{scenario}_2020_2039_3sigma", rsr_prb_folder, rsp_arr_3sigma, rsp_arr_1sigma.shape[1], rsp_arr_1sigma.shape[0], gt)
              
        
                # get mean 2040-2059
                rsp_arr_0sigma = ts_result_0sigma[25: 45].sum(axis=0).astype(np.float32)/ ts_result_0sigma[25: 45].shape[0]
                rsp_arr_1sigma = ts_result_1sigma[25: 45].sum(axis=0).astype(np.float32)/ ts_result_1sigma[25: 45].shape[0]
                rsp_arr_2sigma = ts_result_2sigma[25: 45].sum(axis=0).astype(np.float32)/ ts_result_2sigma[25: 45].shape[0]
                rsp_arr_3sigma = ts_result_3sigma[25: 45].sum(axis=0).astype(np.float32)/ ts_result_3sigma[25: 45].shape[0]
            
                rsp_arr_0sigma[ts_result_0sigma[0]==2] = np.nan
                rsp_arr_1sigma[ts_result_1sigma[0]==2] = np.nan
                rsp_arr_2sigma[ts_result_1sigma[0]==2] = np.nan
                rsp_arr_3sigma[ts_result_1sigma[0]==2] = np.nan
                
                rsp_arr_0sigma[mask_arr] = np.nan
                rsp_arr_1sigma[mask_arr] = np.nan
                rsp_arr_2sigma[mask_arr] = np.nan
                rsp_arr_3sigma[mask_arr] = np.nan
                
                write_image(f"CDHW_RSP_{model}_{scenario}_2040_2059_0sigma", rsr_prb_folder, rsp_arr_0sigma, rsp_arr_0sigma.shape[1], rsp_arr_0sigma.shape[0], gt)
                write_image(f"CDHW_RSP_{model}_{scenario}_2040_2059_1sigma", rsr_prb_folder, rsp_arr_1sigma, rsp_arr_1sigma.shape[1], rsp_arr_1sigma.shape[0], gt)
                write_image(f"CDHW_RSP_{model}_{scenario}_2040_2059_2sigma", rsr_prb_folder, rsp_arr_2sigma, rsp_arr_1sigma.shape[1], rsp_arr_1sigma.shape[0], gt)
                write_image(f"CDHW_RSP_{model}_{scenario}_2040_2059_3sigma", rsr_prb_folder, rsp_arr_3sigma, rsp_arr_1sigma.shape[1], rsp_arr_1sigma.shape[0], gt)
    
                # get mean 2080-2099
                rsp_arr_0sigma = ts_result_0sigma[65: 85].sum(axis=0).astype(np.float32)/ ts_result_0sigma[65: 85].shape[0]
                rsp_arr_1sigma = ts_result_1sigma[65: 85].sum(axis=0).astype(np.float32)/ ts_result_1sigma[65: 85].shape[0]
                rsp_arr_2sigma = ts_result_2sigma[65: 85].sum(axis=0).astype(np.float32)/ ts_result_2sigma[65: 85].shape[0]
                rsp_arr_3sigma = ts_result_3sigma[65: 85].sum(axis=0).astype(np.float32)/ ts_result_3sigma[65: 85].shape[0]
            
                rsp_arr_0sigma[ts_result_0sigma[0]==2] = np.nan
                rsp_arr_1sigma[ts_result_1sigma[0]==2] = np.nan
                rsp_arr_2sigma[ts_result_1sigma[0]==2] = np.nan
                rsp_arr_3sigma[ts_result_1sigma[0]==2] = np.nan
                
                rsp_arr_0sigma[mask_arr] = np.nan
                rsp_arr_1sigma[mask_arr] = np.nan
                rsp_arr_2sigma[mask_arr] = np.nan
                rsp_arr_3sigma[mask_arr] = np.nan
                
                write_image(f"CDHW_RSP_{model}_{scenario}_2080_2099_0sigma", rsr_prb_folder, rsp_arr_0sigma, rsp_arr_0sigma.shape[1], rsp_arr_0sigma.shape[0], gt)
                write_image(f"CDHW_RSP_{model}_{scenario}_2080_2099_1sigma", rsr_prb_folder, rsp_arr_1sigma, rsp_arr_1sigma.shape[1], rsp_arr_1sigma.shape[0], gt)
                write_image(f"CDHW_RSP_{model}_{scenario}_2080_2099_2sigma", rsr_prb_folder, rsp_arr_2sigma, rsp_arr_1sigma.shape[1], rsp_arr_1sigma.shape[0], gt)
                write_image(f"CDHW_RSP_{model}_{scenario}_2080_2099_3sigma", rsr_prb_folder, rsp_arr_3sigma, rsp_arr_1sigma.shape[1], rsp_arr_1sigma.shape[0], gt)
        
