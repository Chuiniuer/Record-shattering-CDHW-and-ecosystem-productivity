# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:12:53 2024

@author: Bohao Li
"""

import numpy as np
import pandas as pd
from osgeo import gdal, osr

from concurrent.futures import ProcessPoolExecutor
# CAPE CIN CIWV VIMC sensible heat flux latent heat flux SH RH SIF GPP


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
    output_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/anomalies/global_mean_warm_seasons/"
    
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
        
        cape_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        cin_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        ciwv_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        vimc_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        sens_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        lat_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        q_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        rh_tmp = np.full((160, out_ysize, out_xsize), np.nan)
        sif_tmp = np.full((100, out_ysize, out_xsize), np.nan)
        gpp_tmp = np.full((100, out_ysize, out_xsize), np.nan)
        
        warm_seasons_ds = gdal.Open(r"/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/warm_seasons.tif")
        warm_seasons = warm_seasons_ds.ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                ysize=out_ysize)
        cape_result = np.full((out_ysize, out_xsize), np.nan)
        cin_result = np.full((out_ysize, out_xsize), np.nan)
        ciwv_result = np.full((out_ysize, out_xsize), np.nan)
        vimc_result = np.full((out_ysize, out_xsize), np.nan)
        sens_result = np.full((out_ysize, out_xsize), np.nan)
        lat_result = np.full((out_ysize, out_xsize), np.nan)
        q_result = np.full((out_ysize, out_xsize), np.nan)
        rh_result = np.full((out_ysize, out_xsize), np.nan)
        sif_result = np.full((out_ysize, out_xsize), np.nan)
        gpp_result = np.full((out_ysize, out_xsize), np.nan)
        
        
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
                        count += 1
                count = 0
                for y in range(20):
                    for mi in m_index_list:
                        sif_tmp[count, i, j] = sif[12*y+mi, i, j]
                        gpp_tmp[count, i, j] = gpp[12*y+mi, i, j]
                        count += 1
                
                #get the mean values
                cape_result[i, j] = cape_tmp[:, i, j].mean()
                cin_result[i, j] = cin_tmp[:, i, j].mean()
                ciwv_result[i, j] = ciwv_tmp[:, i, j].mean()
                vimc_result[i, j] = vimc_tmp[:, i, j].mean()
                sens_result[i, j] = sens_tmp[:, i, j].mean()
                lat_result[i, j] = lat_tmp[:, i, j].mean()
                q_result[i, j] = q_tmp[:, i, j].mean()
                rh_result[i, j] = rh_tmp[:, i, j].mean()
                sif_result[i, j] = sif_tmp[:, i, j].mean()
                gpp_result[i, j] = gpp_tmp[:, i, j].mean()

        write_image(f"cape_{c}_{l}", output_folder, cape_result, out_xsize, out_ysize, new_gt)
        write_image(f"cin_{c}_{l}", output_folder, cin_result, out_xsize, out_ysize, new_gt)
        write_image(f"ciwv_{c}_{l}", output_folder, ciwv_result, out_xsize, out_ysize, new_gt)
        write_image(f"vimc_{c}_{l}", output_folder, vimc_result, out_xsize, out_ysize, new_gt)
        write_image(f"sens_hf_{c}_{l}", output_folder, sens_result, out_xsize, out_ysize, new_gt)
        write_image(f"latent_hf_{c}_{l}", output_folder, lat_result, out_xsize, out_ysize, new_gt)
        write_image(f"Q_{c}_{l}", output_folder, q_result, out_xsize, out_ysize, new_gt)
        write_image(f"RH_{c}_{l}", output_folder, rh_result, out_xsize, out_ysize, new_gt)
        write_image(f"sif_{c}_{l}", output_folder, sif_result, out_xsize, out_ysize, new_gt)
        write_image(f"gpp_{c}_{l}", output_folder, gpp_result, out_xsize, out_ysize, new_gt)

                
if __name__ == "__main__":
    blocky = 40
    blockys = list(range(blocky))
    # for l in blockys:
    #l=10
    #main(l)

    with ProcessPoolExecutor(max_workers=40) as pool:
    	pool.map(main, blockys)
            
                

        
        
        
        
        
        
        