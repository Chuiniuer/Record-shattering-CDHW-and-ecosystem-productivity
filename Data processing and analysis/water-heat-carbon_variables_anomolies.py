# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:15:43 2024

@author: Bohao Li
"""

import numpy as np
import pandas as pd
from osgeo import gdal, osr

from concurrent.futures import ProcessPoolExecutor
# CAPE CIN CIWV VIMC sensible heat flux latent heat flux SH RH

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
    
def main(l):
    CAPE_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/CAPE_converted_resampled/"
    CIN_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/CIN_resampled/"
    CIWV_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/TCWV_resampled/"
    VIMC_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/VIMD_resampled/"
    sensible_hf_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/ERA5_SensHF_W-m-2_Monthly/"
    latent_hf_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/ERA5_LatHF_W-m-2_Monthly/"
    q_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/ERA5_Q_Monthly/"
    RH_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/ERA5_RH_Monthly/"
    SIF_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/CSIF_monthly_GTIFF_resampled/"
    GPP_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/GPP_monthly_GTIFF_resampled/"
    CDHW_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/CDHW_severity/"
    CDHW_RS_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/CDHW_record_shattering_time_series/"
    ws_mean_variables_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/global_mean_merged/"
    
    output_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/data/variable_anomalies/"
    
    ds = gdal.Open(r"/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/warm_seasons.tif")
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    #分块数
    blockx = 20
    blocky = 40
    #分块后每块的大小（除最后一块）
    bxsize = int(xsize / blockx) + 1
    bysize = int(ysize / blocky) + 1
    
    for c in range(blockx):
        # c = 10
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
    
        for y in range(1991, 2023):
            for m in range(1, 13):
                if y==1991 and m==1:
                    cape = np.expand_dims(gdal.Open(CAPE_folder+f"cape_{y}-0{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)
                    cin = np.expand_dims(gdal.Open(CIN_folder+f"cin_{y}-0{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)
                    ciwv = np.expand_dims(gdal.Open(CIWV_folder+f"tcwv_{y}-0{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)
                    vimc = np.expand_dims(gdal.Open(VIMC_folder+f"vimd_{y}-0{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)
                    sens_hf = np.expand_dims(gdal.Open(sensible_hf_folder+f"SensibleHeatFlux_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)
                    lat_hf = np.expand_dims(gdal.Open(latent_hf_folder+f"LatentHeatFlux_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)
                    q =  np.expand_dims(gdal.Open(q_folder+f"ERA5_Q_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)   
                    rh = np.expand_dims(gdal.Open(RH_folder+f"ERA5_rh_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)  
                else:
                    if m < 10:
                        cape = np.concatenate((cape, np.expand_dims(gdal.Open(CAPE_folder+f"cape_{y}-0{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                                ysize=out_ysize), axis=0)), axis=0)
                        cin = np.concatenate((cin, np.expand_dims(gdal.Open(CIN_folder+f"cin_{y}-0{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                                ysize=out_ysize), axis=0)), axis=0)
                        ciwv = np.concatenate((ciwv, np.expand_dims(gdal.Open(CIWV_folder+f"tcwv_{y}-0{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                                ysize=out_ysize), axis=0)), axis=0)
                        vimc = np.concatenate((vimc, np.expand_dims(gdal.Open(VIMC_folder+f"vimd_{y}-0{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                                ysize=out_ysize), axis=0)), axis=0)
                    else:
                        cape = np.concatenate((cape, np.expand_dims(gdal.Open(CAPE_folder+f"cape_{y}-{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                                ysize=out_ysize), axis=0)), axis=0)
                        cin = np.concatenate((cin, np.expand_dims(gdal.Open(CIN_folder+f"cin_{y}-{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                                ysize=out_ysize), axis=0)), axis=0)
                        ciwv = np.concatenate((ciwv, np.expand_dims(gdal.Open(CIWV_folder+f"tcwv_{y}-{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                                ysize=out_ysize), axis=0)), axis=0)
                        vimc = np.concatenate((vimc, np.expand_dims(gdal.Open(VIMC_folder+f"vimd_{y}-{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                                ysize=out_ysize), axis=0)), axis=0)
                    sens_hf = np.concatenate((sens_hf, np.expand_dims(gdal.Open(sensible_hf_folder+f"SensibleHeatFlux_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)), axis=0)
                    lat_hf = np.concatenate((lat_hf, np.expand_dims(gdal.Open(latent_hf_folder+f"LatentHeatFlux_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)), axis=0)
                    q = np.concatenate((q, np.expand_dims(gdal.Open(q_folder+f"ERA5_Q_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)), axis=0)
                    rh = np.concatenate((rh, np.expand_dims(gdal.Open(RH_folder+f"ERA5_rh_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)), axis=0)
            
        #read CDHW events
        
        for y in range(1991, 2023):
            if y==1991:
                cdhw = gdal.Open(CDHW_folder+f"CDHW_seve_{y}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize)
            else:
                cdhw = np.concatenate((cdhw, gdal.Open(CDHW_folder+f"CDHW_seve_{y}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize)), axis=0)
        cdhw[cdhw>0] = 1
        #here the time series are annual        
        cdhw_0sigma = gdal.Open(CDHW_RS_folder+f"CDHW_1991_2022_0sigma.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        cdhw_1sigma = gdal.Open(CDHW_RS_folder+f"CDHW_1991_2022_1sigma.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        cdhw_2sigma = gdal.Open(CDHW_RS_folder+f"CDHW_1991_2022_2sigma.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        cdhw_3sigma = gdal.Open(CDHW_RS_folder+f"CDHW_1991_2022_3sigma.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        
        cdhw_not_rs = cdhw.copy()
        for i in range(cdhw_0sigma.shape[0]):
            for m in range(12):
                cdhw_not_rs[i*12+m][cdhw_0sigma[i]==1] = 0
                
        cdhw_rs_0sigma = cdhw.copy()
        cdhw_rs_1sigma = cdhw.copy()
        cdhw_rs_2sigma = cdhw.copy()
        cdhw_rs_3sigma = cdhw.copy()
        
        for i in range(cdhw_0sigma.shape[0]):
            for m in range(12):
                cdhw_rs_0sigma[i*12+m][cdhw_0sigma[i]==0] = 0
                cdhw_rs_1sigma[i*12+m][cdhw_1sigma[i]==0] = 0
                cdhw_rs_2sigma[i*12+m][cdhw_2sigma[i]==0] = 0
                cdhw_rs_3sigma[i*12+m][cdhw_3sigma[i]==0] = 0
        
        
        #SIF and GPP
        for y in range(2000, 2020):
            for m in range(1, 13):
                if y==2000 and m==1:
                    sif = np.expand_dims(gdal.Open(SIF_folder+f"clear_daily_SIF_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)
                    gpp = np.expand_dims(gdal.Open(GPP_folder+f"GPP_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)
                else:

                    sif = np.concatenate((sif, np.expand_dims(gdal.Open(SIF_folder+f"clear_daily_SIF_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)), axis=0)
                    gpp = np.concatenate((gpp, np.expand_dims(gdal.Open(GPP_folder+f"GPP_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                            ysize=out_ysize), axis=0)), axis=0)
        
        gpp[gpp==-32768] = np.nan
        sens_hf[sens_hf==-32768] = np.nan
        lat_hf[lat_hf==-32768] = np.nan
        cin[np.isnan(cin)] = 0
        print(f"cin_stats: max:{cin.max()}, min:{cin.min()}")
        
        cape_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        cin_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        ciwv_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        vimc_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        sens_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        lat_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        q_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        rh_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        cdhw_atms = np.full((160, out_ysize, out_xsize), np.nan)
        cdhw_not_rs_atms = np.full((160, out_ysize, out_xsize), np.nan)
        cdhw_0sigma_atms = np.full((160, out_ysize, out_xsize), np.nan)
        cdhw_1sigma_atms = np.full((160, out_ysize, out_xsize), np.nan)
        cdhw_2sigma_atms = np.full((160, out_ysize, out_xsize), np.nan)
        cdhw_3sigma_atms = np.full((160, out_ysize, out_xsize), np.nan)
        
        
        sif_tmp = np.full((100, out_ysize, out_xsize), np.nan)
        gpp_tmp = np.full((100, out_ysize, out_xsize), np.nan)
        cdhw_eco = np.full((100, out_ysize, out_xsize), np.nan)
        cdhw_not_rs_eco = np.full((100, out_ysize, out_xsize), np.nan)
        cdhw_0sigma_eco = np.full((100, out_ysize, out_xsize), np.nan)
        cdhw_1sigma_eco = np.full((100, out_ysize, out_xsize), np.nan)
        cdhw_2sigma_eco = np.full((100, out_ysize, out_xsize), np.nan)
        cdhw_3sigma_eco = np.full((100, out_ysize, out_xsize), np.nan)
        
        
        
        warm_seasons_ds = gdal.Open(r"/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/warm_seasons.tif")
        warm_seasons = warm_seasons_ds.ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        
        cape_mean = gdal.Open(ws_mean_variables_folder+"cape.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        cin_mean = gdal.Open(ws_mean_variables_folder+"cin.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        ciwv_mean = gdal.Open(ws_mean_variables_folder+"ciwv.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        vimc_mean = gdal.Open(ws_mean_variables_folder+"vimc.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        sens_mean = gdal.Open(ws_mean_variables_folder+"sens.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        lat_mean = gdal.Open(ws_mean_variables_folder+"latent.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        q_mean = gdal.Open(ws_mean_variables_folder+"q.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        rh_mean = gdal.Open(ws_mean_variables_folder+"RH.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        sif_mean = gdal.Open(ws_mean_variables_folder+"sif.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        gpp_mean = gdal.Open(ws_mean_variables_folder+"gpp.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        
        cin_mean[np.isnan(cin_mean)] = 0
        print(f"cin_mean_stats: max:{cin_mean.max()}, min:{cin_mean.min()}")
        
        gt = warm_seasons_ds.GetGeoTransform()
        for i in range(warm_seasons.shape[0]):
            for j in range(warm_seasons.shape[1]):
                start_month = warm_seasons[i, j]
                print(start_month)
                if start_month == 255:
                    continue
                
                m_index_list = []
                for m in range(5):
                    if m+start_month > 12:
                        m_index = m+start_month-13
    
                    else:
                        m_index = m+start_month-1
                    m_index_list.append(m_index)
                
                #annual iterations
                count = 0
                for y in range(32):
                    for mi in m_index_list:
                        cape_tmp[count, i, j] = cape[12*y+mi, i, j]
                        cin_tmp[count, i, j] = cin[12*y+mi, i, j]
                        ciwv_tmp[count, i, j] = ciwv[12*y+mi, i, j]
                        vimc_tmp[count, i, j] = vimc[12*y+mi, i, j]
                        sens_tmp[count, i, j] = sens_hf[12*y+mi, i, j]
                        lat_tmp[count, i, j] = lat_hf[12*y+mi, i, j]
                        q_tmp[count, i, j] = q[12*y+mi, i, j]
                        rh_tmp[count, i, j] = rh[12*y+mi, i, j]
                        cdhw_atms[count, i, j] = cdhw[12*y+mi, i, j]
                        cdhw_not_rs_atms[count, i, j] = cdhw_not_rs[12*y+mi, i, j]
                        cdhw_0sigma_atms[count, i, j] = cdhw_rs_0sigma[12*y+mi, i, j]
                        cdhw_1sigma_atms[count, i, j] = cdhw_rs_1sigma[12*y+mi, i, j]
                        cdhw_2sigma_atms[count, i, j] = cdhw_rs_2sigma[12*y+mi, i, j]
                        cdhw_3sigma_atms[count, i, j] = cdhw_rs_3sigma[12*y+mi, i, j]
                        count += 1
                count = 0
                
                # from 2000 to 2019
                for y in range(20):
                    for mi in m_index_list:
                        sif_tmp[count, i, j] = sif[12*y+mi, i, j]
                        gpp_tmp[count, i, j] = gpp[12*y+mi, i, j]
                        cdhw_eco[count, i, j] = cdhw[12*(y+9)+mi, i, j]
                        cdhw_not_rs_eco[count, i, j] = cdhw_not_rs[12*(y+9)+mi, i, j]
                        cdhw_0sigma_eco[count, i, j] = cdhw_rs_0sigma[12*(y+9)+mi, i, j]
                        cdhw_1sigma_eco[count, i, j] = cdhw_rs_1sigma[12*(y+9)+mi, i, j]
                        cdhw_2sigma_eco[count, i, j] = cdhw_rs_2sigma[12*(y+9)+mi, i, j]
                        cdhw_3sigma_eco[count, i, j] = cdhw_rs_3sigma[12*(y+9)+mi, i, j]
                        count += 1
        
        # get the anomalies
        #CDHW
        cape_cp = cape_tmp.copy()
        cin_cp = cin_tmp.copy()
        ciwv_cp = ciwv_tmp.copy()
        vimc_cp = vimc_tmp.copy()
        sens_cp = sens_tmp.copy()
        lat_cp = lat_tmp.copy()
        q_cp = q_tmp.copy()
        rh_cp = rh_tmp.copy()
        sif_cp = sif_tmp.copy()
        gpp_cp = gpp_tmp.copy()
        
        
        cape_cp[cdhw_atms!=1] = 0
        cin_cp[cdhw_atms!=1] = 0
        ciwv_cp[cdhw_atms!=1] = 0
        vimc_cp[cdhw_atms!=1] = 0
        sens_cp[cdhw_atms!=1] = 0
        lat_cp[cdhw_atms!=1] = 0
        q_cp[cdhw_atms!=1] = 0
        rh_cp[cdhw_atms!=1] = 0
        sif_cp[cdhw_eco!=1] = 0
        gpp_cp[cdhw_eco!=1] = 0
        
        
        cape_anomalies_cdhw = (cape_cp.sum(axis=0)/(cdhw_atms==1).sum(axis=0))-cape_mean
        cin_anomalies_cdhw = (cin_cp.sum(axis=0)/(cdhw_atms==1).sum(axis=0))-cin_mean
        ciwv_anomalies_cdhw = (ciwv_cp.sum(axis=0)/(cdhw_atms==1).sum(axis=0))-ciwv_mean
        vimc_anomalies_cdhw = (vimc_cp.sum(axis=0)/(cdhw_atms==1).sum(axis=0))-vimc_mean
        sens_anomalies_cdhw = (sens_cp.sum(axis=0)/(cdhw_atms==1).sum(axis=0))-sens_mean
        lat_anomalies_cdhw = (lat_cp.sum(axis=0)/(cdhw_atms==1).sum(axis=0))-lat_mean
        q_anomalies_cdhw = (q_cp.sum(axis=0)/(cdhw_atms==1).sum(axis=0))-q_mean
        rh_anomalies_cdhw = (rh_cp.sum(axis=0)/(cdhw_atms==1).sum(axis=0))-rh_mean
        sif_anomalies_cdhw = (sif_cp.sum(axis=0)/(cdhw_eco==1).sum(axis=0))-sif_mean
        gpp_anomalies_cdhw = (gpp_cp.sum(axis=0)/(cdhw_eco==1).sum(axis=0))-gpp_mean
        
        nodata_mask_cape = (((cdhw_atms==1).sum(axis=0))==0) * (~np.isnan(cape_anomalies_cdhw))
        nodata_mask_cin = (((cdhw_atms==1).sum(axis=0))==0) * (~np.isnan(cin_anomalies_cdhw))
        nodata_mask_ciwv = (((cdhw_atms==1).sum(axis=0))==0) * (~np.isnan(ciwv_anomalies_cdhw))
        nodata_mask_vimc = (((cdhw_atms==1).sum(axis=0))==0) * (~np.isnan(vimc_anomalies_cdhw))
        nodata_mask_sens = (((cdhw_atms==1).sum(axis=0))==0) * (~np.isnan(sens_anomalies_cdhw))
        nodata_mask_lat = (((cdhw_atms==1).sum(axis=0))==0) * (~np.isnan(lat_anomalies_cdhw))
        nodata_mask_q = (((cdhw_atms==1).sum(axis=0))==0) * (~np.isnan(q_anomalies_cdhw))
        nodata_mask_rh = (((cdhw_atms==1).sum(axis=0))==0) * (~np.isnan(rh_anomalies_cdhw))
        nodata_mask_sif = (((cdhw_eco==1).sum(axis=0))==0) * (~np.isnan(sif_anomalies_cdhw))
        nodata_mask_gpp = (((cdhw_eco==1).sum(axis=0))==0) * (~np.isnan(gpp_anomalies_cdhw))
        
        
        cape_anomalies_cdhw[nodata_mask_cape] = np.nan
        cin_anomalies_cdhw[nodata_mask_cin] = np.nan
        ciwv_anomalies_cdhw[nodata_mask_ciwv] = np.nan
        vimc_anomalies_cdhw[nodata_mask_vimc] = np.nan
        sens_anomalies_cdhw[nodata_mask_sens] = np.nan
        lat_anomalies_cdhw[nodata_mask_lat] = np.nan
        q_anomalies_cdhw[nodata_mask_q] = np.nan
        rh_anomalies_cdhw[nodata_mask_rh] = np.nan
        sif_anomalies_cdhw[nodata_mask_sif] = np.nan
        gpp_anomalies_cdhw[nodata_mask_gpp] = np.nan
        
        write_image(f"cape_anomalies_cdhw_{c}_{l}", output_folder, cape_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"cin_anomalies_cdhw_{c}_{l}", output_folder, cin_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"ciwv_anomalies_cdhw_{c}_{l}", output_folder, ciwv_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"vimc_anomalies_cdhw_{c}_{l}", output_folder, vimc_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"sens_anomalies_cdhw_{c}_{l}", output_folder, sens_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"lat_anomalies_cdhw_{c}_{l}", output_folder, lat_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"q_anomalies_cdhw_{c}_{l}", output_folder, q_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"RH_anomalies_cdhw_{c}_{l}", output_folder, rh_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"sif_anomalies_cdhw_{c}_{l}", output_folder, sif_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"gpp_anomalies_cdhw_{c}_{l}", output_folder, gpp_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        
        # CDHW not record shattering events
        cape_cp = cape_tmp.copy()
        cin_cp = cin_tmp.copy()
        ciwv_cp = ciwv_tmp.copy()
        vimc_cp = vimc_tmp.copy()
        sens_cp = sens_tmp.copy()
        lat_cp = lat_tmp.copy()
        q_cp = q_tmp.copy()
        rh_cp = rh_tmp.copy()
        sif_cp = sif_tmp.copy()
        gpp_cp = gpp_tmp.copy()
        
        
        cape_cp[cdhw_not_rs_atms!=1] = 0
        cin_cp[cdhw_not_rs_atms!=1] = 0
        ciwv_cp[cdhw_not_rs_atms!=1] = 0
        vimc_cp[cdhw_not_rs_atms!=1] = 0
        sens_cp[cdhw_not_rs_atms!=1] = 0
        lat_cp[cdhw_not_rs_atms!=1] = 0
        q_cp[cdhw_not_rs_atms!=1] = 0
        rh_cp[cdhw_not_rs_atms!=1] = 0
        sif_cp[cdhw_not_rs_eco!=1] = 0
        gpp_cp[cdhw_not_rs_eco!=1] = 0
        
        
        cape_anomalies_cdhw = (cape_cp.sum(axis=0)/(cdhw_not_rs_atms==1).sum(axis=0))-cape_mean
        cin_anomalies_cdhw = (cin_cp.sum(axis=0)/(cdhw_not_rs_atms==1).sum(axis=0))-cin_mean
        ciwv_anomalies_cdhw = (ciwv_cp.sum(axis=0)/(cdhw_not_rs_atms==1).sum(axis=0))-ciwv_mean
        vimc_anomalies_cdhw = (vimc_cp.sum(axis=0)/(cdhw_not_rs_atms==1).sum(axis=0))-vimc_mean
        sens_anomalies_cdhw = (sens_cp.sum(axis=0)/(cdhw_not_rs_atms==1).sum(axis=0))-sens_mean
        lat_anomalies_cdhw = (lat_cp.sum(axis=0)/(cdhw_not_rs_atms==1).sum(axis=0))-lat_mean
        q_anomalies_cdhw = (q_cp.sum(axis=0)/(cdhw_not_rs_atms==1).sum(axis=0))-q_mean
        rh_anomalies_cdhw = (rh_cp.sum(axis=0)/(cdhw_not_rs_atms==1).sum(axis=0))-rh_mean
        sif_anomalies_cdhw = (sif_cp.sum(axis=0)/(cdhw_not_rs_eco==1).sum(axis=0))-sif_mean
        gpp_anomalies_cdhw = (gpp_cp.sum(axis=0)/(cdhw_not_rs_eco==1).sum(axis=0))-gpp_mean
        
        nodata_mask_cape = (((cdhw_not_rs_atms==1).sum(axis=0))==0) * (~np.isnan(cape_anomalies_cdhw))
        nodata_mask_cin = (((cdhw_not_rs_atms==1).sum(axis=0))==0) * (~np.isnan(cin_anomalies_cdhw))
        nodata_mask_ciwv = (((cdhw_not_rs_atms==1).sum(axis=0))==0) * (~np.isnan(ciwv_anomalies_cdhw))
        nodata_mask_vimc = (((cdhw_not_rs_atms==1).sum(axis=0))==0) * (~np.isnan(vimc_anomalies_cdhw))
        nodata_mask_sens = (((cdhw_not_rs_atms==1).sum(axis=0))==0) * (~np.isnan(sens_anomalies_cdhw))
        nodata_mask_lat = (((cdhw_not_rs_atms==1).sum(axis=0))==0) * (~np.isnan(lat_anomalies_cdhw))
        nodata_mask_q = (((cdhw_not_rs_atms==1).sum(axis=0))==0) * (~np.isnan(q_anomalies_cdhw))
        nodata_mask_rh = (((cdhw_not_rs_atms==1).sum(axis=0))==0) * (~np.isnan(rh_anomalies_cdhw))
        nodata_mask_sif = (((cdhw_not_rs_eco==1).sum(axis=0))==0) * (~np.isnan(sif_anomalies_cdhw))
        nodata_mask_gpp = (((cdhw_not_rs_eco==1).sum(axis=0))==0) * (~np.isnan(gpp_anomalies_cdhw))
        
        cape_anomalies_cdhw[nodata_mask_cape] = np.nan
        cin_anomalies_cdhw[nodata_mask_cin] = np.nan
        ciwv_anomalies_cdhw[nodata_mask_ciwv] = np.nan
        vimc_anomalies_cdhw[nodata_mask_vimc] = np.nan
        sens_anomalies_cdhw[nodata_mask_sens] = np.nan
        lat_anomalies_cdhw[nodata_mask_lat] = np.nan
        q_anomalies_cdhw[nodata_mask_q] = np.nan
        rh_anomalies_cdhw[nodata_mask_rh] = np.nan
        sif_anomalies_cdhw[nodata_mask_sif] = np.nan
        gpp_anomalies_cdhw[nodata_mask_gpp] = np.nan
        
        write_image(f"cape_not_rs_anomalies_cdhw_{c}_{l}", output_folder, cape_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"cin_not_rs_anomalies_cdhw_{c}_{l}", output_folder, cin_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"ciwv_not_rs_anomalies_cdhw_{c}_{l}", output_folder, ciwv_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"vimc_not_rs_anomalies_cdhw_{c}_{l}", output_folder, vimc_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"sens_not_rs_anomalies_cdhw_{c}_{l}", output_folder, sens_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"lat_not_rs_anomalies_cdhw_{c}_{l}", output_folder, lat_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"q_not_rs_anomalies_cdhw_{c}_{l}", output_folder, q_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"RH_not_rs_anomalies_cdhw_{c}_{l}", output_folder, rh_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"sif_not_rs_anomalies_cdhw_{c}_{l}", output_folder, sif_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"gpp_not_rs_anomalies_cdhw_{c}_{l}", output_folder, gpp_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        
        # CDHW margin=0sigma
        cape_cp = cape_tmp.copy()
        cin_cp = cin_tmp.copy()
        ciwv_cp = ciwv_tmp.copy()
        vimc_cp = vimc_tmp.copy()
        sens_cp = sens_tmp.copy()
        lat_cp = lat_tmp.copy()
        q_cp = q_tmp.copy()
        rh_cp = rh_tmp.copy()
        sif_cp = sif_tmp.copy()
        gpp_cp = gpp_tmp.copy()
        
        
        cape_cp[cdhw_0sigma_atms!=1] = 0
        cin_cp[cdhw_0sigma_atms!=1] = 0
        ciwv_cp[cdhw_0sigma_atms!=1] = 0
        vimc_cp[cdhw_0sigma_atms!=1] = 0
        sens_cp[cdhw_0sigma_atms!=1] = 0
        lat_cp[cdhw_0sigma_atms!=1] = 0
        q_cp[cdhw_0sigma_atms!=1] = 0
        rh_cp[cdhw_0sigma_atms!=1] = 0
        sif_cp[cdhw_0sigma_eco!=1] = 0
        gpp_cp[cdhw_0sigma_eco!=1] = 0
        
        
        cape_anomalies_cdhw = (cape_cp.sum(axis=0)/(cdhw_0sigma_atms==1).sum(axis=0))-cape_mean
        cin_anomalies_cdhw = (cin_cp.sum(axis=0)/(cdhw_0sigma_atms==1).sum(axis=0))-cin_mean
        ciwv_anomalies_cdhw = (ciwv_cp.sum(axis=0)/(cdhw_0sigma_atms==1).sum(axis=0))-ciwv_mean
        vimc_anomalies_cdhw = (vimc_cp.sum(axis=0)/(cdhw_0sigma_atms==1).sum(axis=0))-vimc_mean
        sens_anomalies_cdhw = (sens_cp.sum(axis=0)/(cdhw_0sigma_atms==1).sum(axis=0))-sens_mean
        lat_anomalies_cdhw = (lat_cp.sum(axis=0)/(cdhw_0sigma_atms==1).sum(axis=0))-lat_mean
        q_anomalies_cdhw = (q_cp.sum(axis=0)/(cdhw_0sigma_atms==1).sum(axis=0))-q_mean
        rh_anomalies_cdhw = (rh_cp.sum(axis=0)/(cdhw_0sigma_atms==1).sum(axis=0))-rh_mean
        sif_anomalies_cdhw = (sif_cp.sum(axis=0)/(cdhw_0sigma_eco==1).sum(axis=0))-sif_mean
        gpp_anomalies_cdhw = (gpp_cp.sum(axis=0)/(cdhw_0sigma_eco==1).sum(axis=0))-gpp_mean
        
        nodata_mask_cape = (((cdhw_0sigma_atms==1).sum(axis=0))==0) * (~np.isnan(cape_anomalies_cdhw))
        nodata_mask_cin = (((cdhw_0sigma_atms==1).sum(axis=0))==0) * (~np.isnan(cin_anomalies_cdhw))
        nodata_mask_ciwv = (((cdhw_0sigma_atms==1).sum(axis=0))==0) * (~np.isnan(ciwv_anomalies_cdhw))
        nodata_mask_vimc = (((cdhw_0sigma_atms==1).sum(axis=0))==0) * (~np.isnan(vimc_anomalies_cdhw))
        nodata_mask_sens = (((cdhw_0sigma_atms==1).sum(axis=0))==0) * (~np.isnan(sens_anomalies_cdhw))
        nodata_mask_lat = (((cdhw_0sigma_atms==1).sum(axis=0))==0) * (~np.isnan(lat_anomalies_cdhw))
        nodata_mask_q = (((cdhw_0sigma_atms==1).sum(axis=0))==0) * (~np.isnan(q_anomalies_cdhw))
        nodata_mask_rh = (((cdhw_0sigma_atms==1).sum(axis=0))==0) * (~np.isnan(rh_anomalies_cdhw))
        nodata_mask_sif = (((cdhw_0sigma_eco==1).sum(axis=0))==0) * (~np.isnan(sif_anomalies_cdhw))
        nodata_mask_gpp = (((cdhw_0sigma_eco==1).sum(axis=0))==0) * (~np.isnan(gpp_anomalies_cdhw))
        
        cape_anomalies_cdhw[nodata_mask_cape] = np.nan
        cin_anomalies_cdhw[nodata_mask_cin] = np.nan
        ciwv_anomalies_cdhw[nodata_mask_ciwv] = np.nan
        vimc_anomalies_cdhw[nodata_mask_vimc] = np.nan
        sens_anomalies_cdhw[nodata_mask_sens] = np.nan
        lat_anomalies_cdhw[nodata_mask_lat] = np.nan
        q_anomalies_cdhw[nodata_mask_q] = np.nan
        rh_anomalies_cdhw[nodata_mask_rh] = np.nan
        sif_anomalies_cdhw[nodata_mask_sif] = np.nan
        gpp_anomalies_cdhw[nodata_mask_gpp] = np.nan
        
        write_image(f"cape_0sigma_anomalies_cdhw_{c}_{l}", output_folder, cape_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"cin_0sigma_anomalies_cdhw_{c}_{l}", output_folder, cin_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"ciwv_0sigma_anomalies_cdhw_{c}_{l}", output_folder, ciwv_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"vimc_0sigma_anomalies_cdhw_{c}_{l}", output_folder, vimc_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"sens_0sigma_anomalies_cdhw_{c}_{l}", output_folder, sens_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"lat_0sigma_anomalies_cdhw_{c}_{l}", output_folder, lat_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"q_0sigma_anomalies_cdhw_{c}_{l}", output_folder, q_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"RH_0sigma_anomalies_cdhw_{c}_{l}", output_folder, rh_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"sif_0sigma_anomalies_cdhw_{c}_{l}", output_folder, sif_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"gpp_0sigma_anomalies_cdhw_{c}_{l}", output_folder, gpp_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        
        # CDHW margin=1sigma
        cape_cp = cape_tmp.copy()
        cin_cp = cin_tmp.copy()
        ciwv_cp = ciwv_tmp.copy()
        vimc_cp = vimc_tmp.copy()
        sens_cp = sens_tmp.copy()
        lat_cp = lat_tmp.copy()
        q_cp = q_tmp.copy()
        rh_cp = rh_tmp.copy()
        sif_cp = sif_tmp.copy()
        gpp_cp = gpp_tmp.copy()
        
        
        cape_cp[cdhw_1sigma_atms!=1] = 0
        cin_cp[cdhw_1sigma_atms!=1] = 0
        ciwv_cp[cdhw_1sigma_atms!=1] = 0
        vimc_cp[cdhw_1sigma_atms!=1] = 0
        sens_cp[cdhw_1sigma_atms!=1] = 0
        lat_cp[cdhw_1sigma_atms!=1] = 0
        q_cp[cdhw_1sigma_atms!=1] = 0
        rh_cp[cdhw_1sigma_atms!=1] = 0
        sif_cp[cdhw_1sigma_eco!=1] = 0
        gpp_cp[cdhw_1sigma_eco!=1] = 0
        
        
        cape_anomalies_cdhw = (cape_cp.sum(axis=0)/(cdhw_1sigma_atms==1).sum(axis=0))-cape_mean
        cin_anomalies_cdhw = (cin_cp.sum(axis=0)/(cdhw_1sigma_atms==1).sum(axis=0))-cin_mean
        ciwv_anomalies_cdhw = (ciwv_cp.sum(axis=0)/(cdhw_1sigma_atms==1).sum(axis=0))-ciwv_mean
        vimc_anomalies_cdhw = (vimc_cp.sum(axis=0)/(cdhw_1sigma_atms==1).sum(axis=0))-vimc_mean
        sens_anomalies_cdhw = (sens_cp.sum(axis=0)/(cdhw_1sigma_atms==1).sum(axis=0))-sens_mean
        lat_anomalies_cdhw = (lat_cp.sum(axis=0)/(cdhw_1sigma_atms==1).sum(axis=0))-lat_mean
        q_anomalies_cdhw = (q_cp.sum(axis=0)/(cdhw_1sigma_atms==1).sum(axis=0))-q_mean
        rh_anomalies_cdhw = (rh_cp.sum(axis=0)/(cdhw_1sigma_atms==1).sum(axis=0))-rh_mean
        sif_anomalies_cdhw = (sif_cp.sum(axis=0)/(cdhw_1sigma_eco==1).sum(axis=0))-sif_mean
        gpp_anomalies_cdhw = (gpp_cp.sum(axis=0)/(cdhw_1sigma_eco==1).sum(axis=0))-gpp_mean
        
        nodata_mask_cape = (((cdhw_1sigma_atms==1).sum(axis=0))==0) * (~np.isnan(cape_anomalies_cdhw))
        nodata_mask_cin = (((cdhw_1sigma_atms==1).sum(axis=0))==0) * (~np.isnan(cin_anomalies_cdhw))
        nodata_mask_ciwv = (((cdhw_1sigma_atms==1).sum(axis=0))==0) * (~np.isnan(ciwv_anomalies_cdhw))
        nodata_mask_vimc = (((cdhw_1sigma_atms==1).sum(axis=0))==0) * (~np.isnan(vimc_anomalies_cdhw))
        nodata_mask_sens = (((cdhw_1sigma_atms==1).sum(axis=0))==0) * (~np.isnan(sens_anomalies_cdhw))
        nodata_mask_lat = (((cdhw_1sigma_atms==1).sum(axis=0))==0) * (~np.isnan(lat_anomalies_cdhw))
        nodata_mask_q = (((cdhw_1sigma_atms==1).sum(axis=0))==0) * (~np.isnan(q_anomalies_cdhw))
        nodata_mask_rh = (((cdhw_1sigma_atms==1).sum(axis=0))==0) * (~np.isnan(rh_anomalies_cdhw))
        nodata_mask_sif = (((cdhw_1sigma_eco==1).sum(axis=0))==0) * (~np.isnan(sif_anomalies_cdhw))
        nodata_mask_gpp = (((cdhw_1sigma_eco==1).sum(axis=0))==0) * (~np.isnan(gpp_anomalies_cdhw))
        
        cape_anomalies_cdhw[nodata_mask_cape] = np.nan
        cin_anomalies_cdhw[nodata_mask_cin] = np.nan
        ciwv_anomalies_cdhw[nodata_mask_ciwv] = np.nan
        vimc_anomalies_cdhw[nodata_mask_vimc] = np.nan
        sens_anomalies_cdhw[nodata_mask_sens] = np.nan
        lat_anomalies_cdhw[nodata_mask_lat] = np.nan
        q_anomalies_cdhw[nodata_mask_q] = np.nan
        rh_anomalies_cdhw[nodata_mask_rh] = np.nan
        sif_anomalies_cdhw[nodata_mask_sif] = np.nan
        gpp_anomalies_cdhw[nodata_mask_gpp] = np.nan
        write_image(f"cape_1sigma_anomalies_cdhw_{c}_{l}", output_folder, cape_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"cin_1sigma_anomalies_cdhw_{c}_{l}", output_folder, cin_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"ciwv_1sigma_anomalies_cdhw_{c}_{l}", output_folder, ciwv_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"vimc_1sigma_anomalies_cdhw_{c}_{l}", output_folder, vimc_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"sens_1sigma_anomalies_cdhw_{c}_{l}", output_folder, sens_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"lat_1sigma_anomalies_cdhw_{c}_{l}", output_folder, lat_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"q_1sigma_anomalies_cdhw_{c}_{l}", output_folder, q_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"RH_1sigma_anomalies_cdhw_{c}_{l}", output_folder, rh_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"sif_1sigma_anomalies_cdhw_{c}_{l}", output_folder, sif_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"gpp_1sigma_anomalies_cdhw_{c}_{l}", output_folder, gpp_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        
        # CDHW margin=2sigma
        cape_cp = cape_tmp.copy()
        cin_cp = cin_tmp.copy()
        ciwv_cp = ciwv_tmp.copy()
        vimc_cp = vimc_tmp.copy()
        sens_cp = sens_tmp.copy()
        lat_cp = lat_tmp.copy()
        q_cp = q_tmp.copy()
        rh_cp = rh_tmp.copy()
        sif_cp = sif_tmp.copy()
        gpp_cp = gpp_tmp.copy()
        
        
        cape_cp[cdhw_2sigma_atms!=1] = 0
        cin_cp[cdhw_2sigma_atms!=1] = 0
        ciwv_cp[cdhw_2sigma_atms!=1] = 0
        vimc_cp[cdhw_2sigma_atms!=1] = 0
        sens_cp[cdhw_2sigma_atms!=1] = 0
        lat_cp[cdhw_2sigma_atms!=1] = 0
        q_cp[cdhw_2sigma_atms!=1] = 0
        rh_cp[cdhw_2sigma_atms!=1] = 0
        sif_cp[cdhw_2sigma_eco!=1] = 0
        gpp_cp[cdhw_2sigma_eco!=1] = 0
        
        
        cape_anomalies_cdhw = (cape_cp.sum(axis=0)/(cdhw_2sigma_atms==1).sum(axis=0))-cape_mean
        cin_anomalies_cdhw = (cin_cp.sum(axis=0)/(cdhw_2sigma_atms==1).sum(axis=0))-cin_mean
        ciwv_anomalies_cdhw = (ciwv_cp.sum(axis=0)/(cdhw_2sigma_atms==1).sum(axis=0))-ciwv_mean
        vimc_anomalies_cdhw = (vimc_cp.sum(axis=0)/(cdhw_2sigma_atms==1).sum(axis=0))-vimc_mean
        sens_anomalies_cdhw = (sens_cp.sum(axis=0)/(cdhw_2sigma_atms==1).sum(axis=0))-sens_mean
        lat_anomalies_cdhw = (lat_cp.sum(axis=0)/(cdhw_2sigma_atms==1).sum(axis=0))-lat_mean
        q_anomalies_cdhw = (q_cp.sum(axis=0)/(cdhw_2sigma_atms==1).sum(axis=0))-q_mean
        rh_anomalies_cdhw = (rh_cp.sum(axis=0)/(cdhw_2sigma_atms==1).sum(axis=0))-rh_mean
        sif_anomalies_cdhw = (sif_cp.sum(axis=0)/(cdhw_2sigma_eco==1).sum(axis=0))-sif_mean
        gpp_anomalies_cdhw = (gpp_cp.sum(axis=0)/(cdhw_2sigma_eco==1).sum(axis=0))-gpp_mean
        
        
        nodata_mask_cape = (((cdhw_2sigma_atms==1).sum(axis=0))==0) * (~np.isnan(cape_anomalies_cdhw))
        nodata_mask_cin = (((cdhw_2sigma_atms==1).sum(axis=0))==0) * (~np.isnan(cin_anomalies_cdhw))
        nodata_mask_ciwv = (((cdhw_2sigma_atms==1).sum(axis=0))==0) * (~np.isnan(ciwv_anomalies_cdhw))
        nodata_mask_vimc = (((cdhw_2sigma_atms==1).sum(axis=0))==0) * (~np.isnan(vimc_anomalies_cdhw))
        nodata_mask_sens = (((cdhw_2sigma_atms==1).sum(axis=0))==0) * (~np.isnan(sens_anomalies_cdhw))
        nodata_mask_lat = (((cdhw_2sigma_atms==1).sum(axis=0))==0) * (~np.isnan(lat_anomalies_cdhw))
        nodata_mask_q = (((cdhw_2sigma_atms==1).sum(axis=0))==0) * (~np.isnan(q_anomalies_cdhw))
        nodata_mask_rh = (((cdhw_2sigma_atms==1).sum(axis=0))==0) * (~np.isnan(rh_anomalies_cdhw))
        nodata_mask_sif = (((cdhw_2sigma_eco==1).sum(axis=0))==0) * (~np.isnan(sif_anomalies_cdhw))
        nodata_mask_gpp = (((cdhw_2sigma_eco==1).sum(axis=0))==0) * (~np.isnan(gpp_anomalies_cdhw))
        
        cape_anomalies_cdhw[nodata_mask_cape] = np.nan
        cin_anomalies_cdhw[nodata_mask_cin] = np.nan
        ciwv_anomalies_cdhw[nodata_mask_ciwv] = np.nan
        vimc_anomalies_cdhw[nodata_mask_vimc] = np.nan
        sens_anomalies_cdhw[nodata_mask_sens] = np.nan
        lat_anomalies_cdhw[nodata_mask_lat] = np.nan
        q_anomalies_cdhw[nodata_mask_q] = np.nan
        rh_anomalies_cdhw[nodata_mask_rh] = np.nan
        sif_anomalies_cdhw[nodata_mask_sif] = np.nan
        gpp_anomalies_cdhw[nodata_mask_gpp] = np.nan
        
        write_image(f"cape_2sigma_anomalies_cdhw_{c}_{l}", output_folder, cape_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"cin_2sigma_anomalies_cdhw_{c}_{l}", output_folder, cin_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"ciwv_2sigma_anomalies_cdhw_{c}_{l}", output_folder, ciwv_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"vimc_2sigma_anomalies_cdhw_{c}_{l}", output_folder, vimc_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"sens_2sigma_anomalies_cdhw_{c}_{l}", output_folder, sens_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"lat_2sigma_anomalies_cdhw_{c}_{l}", output_folder, lat_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"q_2sigma_anomalies_cdhw_{c}_{l}", output_folder, q_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"RH_2sigma_anomalies_cdhw_{c}_{l}", output_folder, rh_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"sif_2sigma_anomalies_cdhw_{c}_{l}", output_folder, sif_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"gpp_2sigma_anomalies_cdhw_{c}_{l}", output_folder, gpp_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        
        # CDHW margin=0sigma
        cape_cp = cape_tmp.copy()
        cin_cp = cin_tmp.copy()
        ciwv_cp = ciwv_tmp.copy()
        vimc_cp = vimc_tmp.copy()
        sens_cp = sens_tmp.copy()
        lat_cp = lat_tmp.copy()
        q_cp = q_tmp.copy()
        rh_cp = rh_tmp.copy()
        sif_cp = sif_tmp.copy()
        gpp_cp = gpp_tmp.copy()
        
        
        cape_cp[cdhw_3sigma_atms!=1] = 0
        cin_cp[cdhw_3sigma_atms!=1] = 0
        ciwv_cp[cdhw_3sigma_atms!=1] = 0
        vimc_cp[cdhw_3sigma_atms!=1] = 0
        sens_cp[cdhw_3sigma_atms!=1] = 0
        lat_cp[cdhw_3sigma_atms!=1] = 0
        q_cp[cdhw_3sigma_atms!=1] = 0
        rh_cp[cdhw_3sigma_atms!=1] = 0
        sif_cp[cdhw_3sigma_eco!=1] = 0
        gpp_cp[cdhw_3sigma_eco!=1] = 0
        
        
        cape_anomalies_cdhw = (cape_cp.sum(axis=0)/(cdhw_3sigma_atms==1).sum(axis=0))-cape_mean
        cin_anomalies_cdhw = (cin_cp.sum(axis=0)/(cdhw_3sigma_atms==1).sum(axis=0))-cin_mean
        ciwv_anomalies_cdhw = (ciwv_cp.sum(axis=0)/(cdhw_3sigma_atms==1).sum(axis=0))-ciwv_mean
        vimc_anomalies_cdhw = (vimc_cp.sum(axis=0)/(cdhw_3sigma_atms==1).sum(axis=0))-vimc_mean
        sens_anomalies_cdhw = (sens_cp.sum(axis=0)/(cdhw_3sigma_atms==1).sum(axis=0))-sens_mean
        lat_anomalies_cdhw = (lat_cp.sum(axis=0)/(cdhw_3sigma_atms==1).sum(axis=0))-lat_mean
        q_anomalies_cdhw = (q_cp.sum(axis=0)/(cdhw_3sigma_atms==1).sum(axis=0))-q_mean
        rh_anomalies_cdhw = (rh_cp.sum(axis=0)/(cdhw_3sigma_atms==1).sum(axis=0))-rh_mean
        sif_anomalies_cdhw = (sif_cp.sum(axis=0)/(cdhw_3sigma_eco==1).sum(axis=0))-sif_mean
        gpp_anomalies_cdhw = (gpp_cp.sum(axis=0)/(cdhw_3sigma_eco==1).sum(axis=0))-gpp_mean
        
        nodata_mask_cape = (((cdhw_3sigma_atms==1).sum(axis=0))==0) * (~np.isnan(cape_anomalies_cdhw))
        nodata_mask_cin = (((cdhw_3sigma_atms==1).sum(axis=0))==0) * (~np.isnan(cin_anomalies_cdhw))
        nodata_mask_ciwv = (((cdhw_3sigma_atms==1).sum(axis=0))==0) * (~np.isnan(ciwv_anomalies_cdhw))
        nodata_mask_vimc = (((cdhw_3sigma_atms==1).sum(axis=0))==0) * (~np.isnan(vimc_anomalies_cdhw))
        nodata_mask_sens = (((cdhw_3sigma_atms==1).sum(axis=0))==0) * (~np.isnan(sens_anomalies_cdhw))
        nodata_mask_lat = (((cdhw_3sigma_atms==1).sum(axis=0))==0) * (~np.isnan(lat_anomalies_cdhw))
        nodata_mask_q = (((cdhw_3sigma_atms==1).sum(axis=0))==0) * (~np.isnan(q_anomalies_cdhw))
        nodata_mask_rh = (((cdhw_3sigma_atms==1).sum(axis=0))==0) * (~np.isnan(rh_anomalies_cdhw))
        nodata_mask_sif = (((cdhw_3sigma_eco==1).sum(axis=0))==0) * (~np.isnan(sif_anomalies_cdhw))
        nodata_mask_gpp = (((cdhw_3sigma_eco==1).sum(axis=0))==0) * (~np.isnan(gpp_anomalies_cdhw))
        
        cape_anomalies_cdhw[nodata_mask_cape] = np.nan
        cin_anomalies_cdhw[nodata_mask_cin] = np.nan
        ciwv_anomalies_cdhw[nodata_mask_ciwv] = np.nan
        vimc_anomalies_cdhw[nodata_mask_vimc] = np.nan
        sens_anomalies_cdhw[nodata_mask_sens] = np.nan
        lat_anomalies_cdhw[nodata_mask_lat] = np.nan
        q_anomalies_cdhw[nodata_mask_q] = np.nan
        rh_anomalies_cdhw[nodata_mask_rh] = np.nan
        sif_anomalies_cdhw[nodata_mask_sif] = np.nan
        gpp_anomalies_cdhw[nodata_mask_gpp] = np.nan
        
        write_image(f"cape_3sigma_anomalies_cdhw_{c}_{l}", output_folder, cape_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"cin_3sigma_anomalies_cdhw_{c}_{l}", output_folder, cin_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"ciwv_3sigma_anomalies_cdhw_{c}_{l}", output_folder, ciwv_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"vimc_3sigma_anomalies_cdhw_{c}_{l}", output_folder, vimc_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"sens_3sigma_anomalies_cdhw_{c}_{l}", output_folder, sens_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"lat_3sigma_anomalies_cdhw_{c}_{l}", output_folder, lat_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"q_3sigma_anomalies_cdhw_{c}_{l}", output_folder, q_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"RH_3sigma_anomalies_cdhw_{c}_{l}", output_folder, rh_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"sif_3sigma_anomalies_cdhw_{c}_{l}", output_folder, sif_anomalies_cdhw, out_xsize, out_ysize, new_gt)
        write_image(f"gpp_3sigma_anomalies_cdhw_{c}_{l}", output_folder, gpp_anomalies_cdhw, out_xsize, out_ysize, new_gt)
                


                
if __name__ == "__main__":
    blocky = 40
    blockys = list(range(blocky))
    # for l in blockys:
    # l=10
    # main(l)

    with ProcessPoolExecutor(max_workers=40) as pool:
     	pool.map(main, blockys)