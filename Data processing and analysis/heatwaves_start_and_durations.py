# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:18:29 2024

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

def write_2d_tif(variable_name, output_folder, year, array, geotransform):
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = output_folder + variable_name + '_'  + str(year) + '.tif'
    out_tif = driver.Create(out_tif_name, array.shape[1], array.shape[0], 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])

    # Setting the image display range
    out_tif.SetGeoTransform(geotransform)

    # Get geographic coordinate system
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # Define the output coordinate system as WGS84
    out_tif.SetProjection(srs.ExportToWkt())  # Creating projection information for new layers

    # writing the results
    out_tif.GetRasterBand(1).WriteArray(array)
    out_tif.GetRasterBand(1).SetNoDataValue(255)
    out_tif.FlushCache()
    # print(f'output successfully')
    del out_tif

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
    
def unit_convert_tas(array):
    """
    K -> ℃
    :param array:
    :return:
    """

    array[array==-32768] = -10000
    array = array - 273.15
    array[array<-9999] = -9999
    # array = array.astype(np.int16)
    return array

class IndexExtreme:
    def __init__(self):
        self.days_count = 365
        self.year = 1950
        pass


    def get_heatwave_begin(self, time_series):

        #Here next_year_begin_heat is used to characterise whether or not the end-of-year heatwave bridges to the beginning of the second year, with 1 meaning yes and 0 meaning no, and if it's 1, the first heatwave of the second year needs to be removed
        next_year_begin_heat = 0
        if time_series[0] == 255:
            return time_series[:self.days_count], 0
        # Variables for counting the number of heat wave events and for counting whether or not a heat wave counts as a heat wave
        heat_wave_begin = 0
        heat_days = 0
        count = 0
        result = np.zeros((self.days_count,), np.uint8)
        # Determine if there was a heat wave at the beginning of the year, default no
        for t in range(len(time_series)):

            if t != self.days_count-1:
                if time_series[t] == 1:
                    if count == 0:
                        heat_wave_begin = t
                        count += 1
                    heat_days += 1
                else:
                    if heat_days >= 3:
                        result[heat_wave_begin] = heat_days

                    count = 0
                    heat_days = 0
            else:
                if self.year == 2022:
                    result[-1] = 0
                    break
                else:
                    if time_series[t] == 0:
                        if heat_days >= 3:
                            result[heat_wave_begin] = heat_days
                            count = 0
                        heat_days = 0
                        break
                    else:
                        if count == 0:
                            heat_wave_begin = t
                            count += 1
                        heat_days += 1
                        for tn in range(t+1, t+3):
                            if time_series[tn] == 1:
                                heat_days += 1
                            else:
                                heat_days = 0
                        if heat_days >= 3:
                            result[heat_wave_begin] = heat_days

                        if time_series[t+1] + time_series[t+2] + time_series[t+3] == 3:
                            next_year_begin_heat = 1
                        else:
                            next_year_begin_heat = 0
                        break
        # Actually, here it means that an event is only kicked out if the last day is high temperature and the following three days are consecutively high temperature
        return result, next_year_begin_heat

    def get_heat_wave(self, input_folder, year, out_folder, th):
        # Note that here th is a two-dimensional array
        # try:
            
        for m in range(1, 13):
            if m == 1:
                temp_array = gdal.Open(input_folder + f"ERA5_T2mMax_Daily_{year}_{m}.tif").ReadAsArray()
            else:
                temp_array = np.concatenate((temp_array, gdal.Open(input_folder + f"ERA5_T2mMax_Daily_{year}_{m}.tif").ReadAsArray()), axis=0)
        temp_array = unit_convert_tas(temp_array)
        th1 = np.repeat(th[np.newaxis, :, :], temp_array.shape[0], axis=0)
        array = (temp_array >= th1).astype(np.uint8)
        array[temp_array==-9999]=255
        del temp_array
        self.year = year
        self.days_count = array.shape[0]

        if year != 2022:
            
            for m in range(1, 13):
                if m == 1:
                    next_tmp_array = gdal.Open(input_folder + f"ERA5_T2mMax_Daily_{year+1}_{m}.tif").ReadAsArray()
                else:
                    next_tmp_array = np.concatenate((next_tmp_array, gdal.Open(input_folder + f"ERA5_T2mMax_Daily_{year+1}_{m}.tif").ReadAsArray()), axis=0)
            
            next_tmp_array = unit_convert_tas(next_tmp_array)
            if next_tmp_array.shape[0] != array.shape[0]:
                th1 = np.repeat(th[np.newaxis, :, :], next_tmp_array.shape[0], axis=0)
            next_array = (next_tmp_array >= th1).astype(np.uint8)
            next_array[next_tmp_array==-9999] = 255
            del next_tmp_array
            array = np.concatenate((array, next_array), axis=0)

        next_year_begin_heat = np.zeros((array.shape[1], array.shape[2]), dtype=np.uint8)
        temp_result = np.zeros((self.days_count, array.shape[1], array.shape[2]), dtype=np.uint8)
        for i in tqdm(range(array.shape[1])):
            for j in range(array.shape[2]):
                temp_result[:, i, j], next_year_begin_heat[i, j] = self.get_heatwave_begin(array[:, i, j])
        
        gt = gdal.Open(input_folder+"ERA5_T2mMax_Daily_1950_1.tif").GetGeoTransform()
        
        write_3d_tif("HeatWave", out_folder, year, temp_result, gt)
        write_2d_tif("NextYearBeginHeat", out_folder, year+1, next_year_begin_heat, gt)
        # except:
        #     df.loc[num, "data"] = f"HW_{mode}_{scenario}_{year}"
        #     num += 1
        #     df.to_csv(r"G:\HW_check.csv")
        
def mission(y):
    index = IndexExtreme()
    input_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_Tmax_daily/"
    out_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/heatwave_begin/"
    threshold_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/"
    threshold_path = threshold_folder + f"90th_era5.tif"
    threshold = gdal.Open(threshold_path).ReadAsArray()

    index.get_heat_wave(input_folder, y, out_folder, threshold)
        
if __name__ == "__main__":
    years = range(1951, 2023)
    
    with ProcessPoolExecutor(max_workers=20) as pool:
        pool.map(mission, years)







