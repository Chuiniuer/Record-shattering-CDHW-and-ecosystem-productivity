# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:48:23 2024

@author: Bohao Li
"""

from osgeo import gdal, osr
import numpy as np
import pandas as pd
import os 

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---")
		print("---  OK  ---")
 

def e_sat(T):
    """
    calculate the satruated vapour expressure

    Parameters
    ----------
    T : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    es0 = 611
    T0 = 273.16
    Lv = 2.5e6
    Rv = 461
    return es0*np.exp(Lv/Rv*(1/T0-1/T))

def specific_humidity(esat_tdew, ps):
    q = 0.622 * esat_tdew / (ps - 0.378*esat_tdew)
    return q
    
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
    
    dwt_input_folder = "G:\\ERA5-land\\ERA5_DT2m_Monthly\\"
    t2m_input_folder = "G:\\ERA5-land\\ERA5_tas_Monthly\\"
    esat_T_folder = "G:\\ERA5-land\\ERA5_esatT_Monthly\\"
    esat_Tdew_folder = "G:\\ERA5-land\\ERA5_esatTdew_Monthly\\"
    ps_folder = "G:\\ERA5-land\\ERA5_SPRESSURE_Monthly\\"
    RH_folder = "G:\\ERA5-land\\ERA5_RH_Monthly\\"
    q_folder = "G:\\ERA5-land\\ERA5_Q_Monthly\\"
    mkdir(esat_Tdew_folder)
    mkdir(esat_T_folder)
    mkdir(RH_folder)
    mkdir(q_folder)
    
    gt = gdal.Open(r"G:\ERA5-land\ERA5_tas_Monthly\ERA5_tas_Monthly_1950_9.tif").GetGeoTransform()
    for y in range(1951, 2023):
        for m in range(1, 13):
            arr = gdal.Open(t2m_input_folder+f"ERA5_tas_Monthly_{y}_{m}.tif").ReadAsArray()
            nodata = (arr==-32768)
            arr[arr==-32768] = np.nan
            esat_t = e_sat(arr)
            write_image(f"ERA5_esatT2m_Monthly_{y}_{m}", esat_T_folder, esat_t, esat_t.shape[1], esat_t.shape[0], gt)
            
            arr = gdal.Open(dwt_input_folder+f"ERA5_DT2m_Monthly_{y}_{m}.tif").ReadAsArray()
            nodata = (arr==-32768)
            arr[arr==-32768] = np.nan
            esat_tdew = e_sat(arr)
            write_image(f"ERA5_esatTdew_Monthly_{y}_{m}", esat_Tdew_folder, esat_tdew, esat_tdew.shape[1], esat_tdew.shape[0], gt)
            
            rh = esat_tdew / esat_t
            write_image(f"ERA5_rh_Monthly_{y}_{m}", RH_folder, rh, rh.shape[1], rh.shape[0], gt)
    
            ps = gdal.Open(ps_folder+f"ERA5__Monthly_{y}_{m}.tif").ReadAsArray()
            ps[ps==-32768] = np.nan
            q = specific_humidity(esat_tdew, ps)
            
            write_image(f"ERA5_Q_Monthly_{y}_{m}", q_folder, q, q.shape[1], q.shape[0], gt)
    
    
    
    
    
    