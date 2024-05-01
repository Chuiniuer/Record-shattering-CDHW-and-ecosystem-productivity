# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:36:14 2024

@author: Bohao Li
"""

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from osgeo import gdal, osr

from concurrent.futures import ProcessPoolExecutor

def sensitivity_analysis():
    pass

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
    output_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_R_square/"
    CDHW_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/CDHW_severity/"
    pr_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_pr_Monthly/"
    ps_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_SPRESSURE_Monthly/"
    tas_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_tas_Monthly/"
    rsds_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_rsds_Monthly/"
    rlds_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_rlds_Monthly/"
    tdew_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_DT2m_Monthly/"
    wind_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_wind_Monthly/"
    rh_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_RH_Monthly/"
    q_folder = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_Q_Monthly/"
    warm_season_path = "/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/warm_seasons.tif"
    ds = gdal.Open(r"/work/home/bjsfdxlbh/Bohao/ecosystem_productivity/era5-land/ERA5_land_stdev/DT2m_std.tif")
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
        
        warm_season = gdal.Open(warm_season_path).ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize)
        
        for y in range(1952, 2023):
            if y==1952:
                CDHW = gdal.Open(CDHW_folder + f"CDHW_seve_{y}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize)
            else:
                CDHW = np.concatenate((CDHW, gdal.Open(CDHW_folder + f"CDHW_seve_{y}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize)), axis=0)
        
        
        for y in range(1952, 2023):
            for m in range(1, 13):
                if y==1952 and m==1:
                    pr = np.expand_dims(gdal.Open(pr_folder+f"ERA5_PrTotal_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)
                    ps = np.expand_dims(gdal.Open(ps_folder+f"ERA5__Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)
                    tas = np.expand_dims(gdal.Open(tas_folder+f"ERA5_tas_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)
                    rsds = np.expand_dims(gdal.Open(rsds_folder+f"ERA5_rsds_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)
                    rlds = np.expand_dims(gdal.Open(rlds_folder+f"ERA5_rlds_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)
                    tdew = np.expand_dims(gdal.Open(tdew_folder+f"ERA5_DT2m_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)
                    rh = np.expand_dims(gdal.Open(rh_folder+f"ERA5_rh_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)
                    wind = np.expand_dims(gdal.Open(wind_folder+f"ERA5_wind_Monthly_{y}_{m}.tif.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)
                    q = np.expand_dims(gdal.Open(q_folder+f"ERA5_Q_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)
                else:
                    pr = np.concatenate((pr, np.expand_dims(gdal.Open(pr_folder+f"ERA5_PrTotal_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)), axis=0)
                    ps = np.concatenate((ps, np.expand_dims(gdal.Open(ps_folder+f"ERA5__Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)), axis=0)
                    tas = np.concatenate((tas, np.expand_dims(gdal.Open(tas_folder+f"ERA5_tas_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)), axis=0)
                    rsds = np.concatenate((rsds, np.expand_dims(gdal.Open(rsds_folder+f"ERA5_rsds_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)), axis=0)
                    rlds = np.concatenate((rlds, np.expand_dims(gdal.Open(rlds_folder+f"ERA5_rlds_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)), axis=0)
                    tdew = np.concatenate((tdew, np.expand_dims(gdal.Open(tdew_folder+f"ERA5_DT2m_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)), axis=0)
                    rh = np.concatenate((rh, np.expand_dims(gdal.Open(rh_folder+f"ERA5_rh_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)), axis=0)
                    wind = np.concatenate((wind, np.expand_dims(gdal.Open(wind_folder+f"ERA5_wind_Monthly_{y}_{m}.tif.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)), axis=0)
                    q = np.concatenate((q, np.expand_dims(gdal.Open(q_folder+f"ERA5_Q_Monthly_{y}_{m}.tif").ReadAsArray(xoff=bxsize*c, yoff=bysize*l, xsize=out_xsize,
                                                        ysize=out_ysize), axis=0)), axis=0)

        
        r_square = np.zeros((out_ysize, out_xsize))
        pr_sens = np.zeros((out_ysize, out_xsize))
        ps_sens = np.zeros((out_ysize, out_xsize))
        tas_sens = np.zeros((out_ysize, out_xsize))
        rsds_sens = np.zeros((out_ysize, out_xsize))
        rlds_sens = np.zeros((out_ysize, out_xsize))
        tdew_sens = np.zeros((out_ysize, out_xsize))
        rh_sens = np.zeros((out_ysize, out_xsize))
        wind_sens = np.zeros((out_ysize, out_xsize))
        Q_sens = np.zeros((out_ysize, out_xsize))
        
        pr_tmp = np.full((360, out_ysize, out_xsize), np.nan)
        ps_tmp = np.full((360, out_ysize, out_xsize), np.nan)
        tas_tmp = np.full((360, out_ysize, out_xsize), np.nan)
        rsds_tmp = np.full((360, out_ysize, out_xsize), np.nan)
        rlds_tmp = np.full((360, out_ysize, out_xsize), np.nan)
        tdew_tmp = np.full((360, out_ysize, out_xsize), np.nan)
        rh_tmp = np.full((360, out_ysize, out_xsize), np.nan)
        wind_tmp = np.full((360, out_ysize, out_xsize), np.nan)
        Q_tmp = np.full((360, out_ysize, out_xsize), np.nan)
        CDHW_tmp = np.full((360, out_ysize, out_xsize), np.nan)
        for i in range(out_ysize):
            for j in range(out_xsize):
                print(warm_season[i, j])
                if warm_season[i, j]==255:
                    r_square[i, j] = np.nan
                    pr_sens[i, j] = np.nan
                    ps_sens[i, j] = np.nan
                    tas_sens[i, j] = np.nan
                    rsds_sens[i, j] = np.nan
                    rlds_sens[i, j] = np.nan
                    tdew_sens[i, j] = np.nan
                    rh_sens[i, j] = np.nan
                    wind_sens[i, j] = np.nan
                    Q_sens[i, j] = np.nan
                    continue
                else:
                    start_month = warm_season[i, j]
                    count = 0
                    
                    m_index_list = []
                    for m in range(5):
                        if m+start_month > 12:
                            m_index = m+start_month-13
        
                        else:
                            m_index = m+start_month-1
                        m_index_list.append(m_index)
                    #annual iterations
                    for y in range(71):
                        for mi in m_index_list:
                            pr_tmp[count, i, j] = pr[12*y+mi, i, j]
                            ps_tmp[count, i, j] = ps[12*y+mi, i, j]
                            tas_tmp[count, i, j] = tas[12*y+mi, i, j]
                            rsds_tmp[count, i, j] = rsds[12*y+mi, i, j]
                            rlds_tmp[count, i, j] = rlds[12*y+mi, i, j]
                            tdew_tmp[count, i, j] = tdew[12*y+mi, i, j]
                            rh_tmp[count, i, j] = rh[12*y+mi, i, j]
                            wind_tmp[count, i, j] = wind[12*y+mi, i, j]
                            Q_tmp[count, i, j] = q[12*y+mi, i, j]
                            CDHW_tmp[count, i, j] = CDHW[12*y+mi, i, j]
                            count += 1
                    CDHW_tmp[CDHW_tmp==0] = np.nan
                    
                    
                    df = pd.DataFrame(columns=["pr", "ps", "t2m", "rsds", "rlds", "tdew", "wind", "rh", "q", "CDHW"])
                    df["pr"] = pr_tmp[:, i, j]
                    df["ps"] = ps_tmp[:, i, j]
                    df["t2m"] = tas_tmp[:, i, j]
                    df["rsds"] = rsds_tmp[:, i, j]
                    df["rlds"] = rlds_tmp[:, i, j]
                    df["tdew"] = tdew_tmp[:, i, j]
                    df["rh"] = rh_tmp[:, i, j]
                    df["wind"] = wind_tmp[:, i, j]
                    df["q"] = Q_tmp[:, i, j]
                    df["CDHW"] = CDHW_tmp[:, i, j]
                    for column in list(df.columns[df.isnull().sum() > 0]):
                        mean_val = df[column].mean()
                        df[column].fillna(mean_val, inplace=True)
                    df["CDHW"] = CDHW_tmp[:, i, j]
                    
                    df.dropna(inplace=True)
                    if len(df) <= 10:
                        r_square[i, j] = np.nan
                        pr_sens[i, j] = np.nan
                        ps_sens[i, j] = np.nan
                        tas_sens[i, j] = np.nan
                        rsds_sens[i, j] = np.nan
                        rlds_sens[i, j] = np.nan
                        tdew_sens[i, j] = np.nan
                        rh_sens[i, j] = np.nan
                        wind_sens[i, j] = np.nan
                        Q_sens[i, j] = np.nan
                        continue
                    

                    CDHW_std = np.array(df["CDHW"]).std()
                    pr_std = np.array(df["pr"]).std()
                    ps_std = np.array(df["ps"]).std()
                    t2m_std = np.array(df["t2m"]).std()
                    rsds_std = np.array(df["rsds"]).std()
                    rlds_std = np.array(df["rlds"]).std()
                    tdew_std = np.array(df["tdew"]).std()
                    rh_std = np.array(df["rh"]).std()
                    wind_std = np.array(df["wind"]).std()
                    Q_std = np.array(df["q"]).std()
                    rfr = RandomForestRegressor(n_estimators=200, random_state=0, oob_score=True)
                    x = df.iloc[:, :-1]
                    print(np.array(x).shape)
                    y = df["CDHW"]
                    rfr.fit(x, y)
                    #pr
                    temp_x = x.copy()
                    temp_x["pr"] = x["pr"]+pr_std
                    cdhws_rf1sd = rfr.predict(temp_x)
                    sc_pr = -(np.array(df["CDHW"])-cdhws_rf1sd).mean()/CDHW_std
                    pr_sens[i, j] = sc_pr
                    #snow
                    temp_x = x.copy()
                    temp_x["ps"] = x["ps"]+ps_std
                    cdhws_rf1sd = rfr.predict(temp_x)
                    sc_ps = -(np.array(df["CDHW"])-cdhws_rf1sd).mean()/CDHW_std
                    ps_sens[i, j] = sc_ps
                    #tas

                    temp_x = x.copy()
                    temp_x["t2m"] = x["t2m"]+t2m_std
                    cdhws_rf1sd = rfr.predict(temp_x)
                    sc_tas = -(np.array(df["CDHW"])-cdhws_rf1sd).mean()/CDHW_std
                    tas_sens[i, j] = sc_tas
                    #rsds
                    temp_x = x.copy()
                    temp_x["rsds"] = x["rsds"]+rsds_std
                    cdhws_rf1sd = rfr.predict(temp_x)
                    sc_rsds = -(np.array(df["CDHW"])-cdhws_rf1sd).mean()/CDHW_std
                    rsds_sens[i, j] = sc_rsds
                    #rlds
                    temp_x = x.copy()
                    temp_x["rlds"] = x["rlds"]+rlds_std
                    cdhws_rf1sd = rfr.predict(temp_x)
                    sc_rlds = -(np.array(df["CDHW"])-cdhws_rf1sd).mean()/CDHW_std
                    rlds_sens[i, j] = sc_rlds
                    #tdew
                    temp_x = x.copy()
                    temp_x["tdew"] = x["tdew"]+tdew_std
                    cdhws_rf1sd = rfr.predict(temp_x)
                    sc_tdew = -(np.array(df["CDHW"])-cdhws_rf1sd).mean()/CDHW_std
                    tdew_sens[i, j] = sc_tdew
                    #rh
                    temp_x = x.copy()
                    temp_x["rh"] = x["rh"]+rh_std
                    cdhws_rf1sd = rfr.predict(temp_x)
                    sc_rh = -(np.array(df["CDHW"])-cdhws_rf1sd).mean()/CDHW_std
                    rh_sens[i, j] = sc_rh
                    #wind
                    temp_x = x.copy()
                    temp_x["wind"] = x["wind"]+wind_std
                    cdhws_rf1sd = rfr.predict(temp_x)
                    sc_wind = -(np.array(df["CDHW"])-cdhws_rf1sd).mean()/CDHW_std
                    wind_sens[i, j] = sc_wind
                    #q
                    temp_x = x.copy()
                    temp_x["q"] = x["q"]+Q_std
                    cdhws_rf1sd = rfr.predict(temp_x)
                    sc_q = -(np.array(df["CDHW"])-cdhws_rf1sd).mean()/CDHW_std
                    Q_sens[i, j] = sc_q
                    
                    
                    print(rfr.oob_score_)
                    r_square[i, j] = rfr.oob_score_      
        write_image(f"R_square_{c}_{l}", output_folder, r_square, out_xsize, out_ysize, new_gt)
        write_image(f"pr_{c}_{l}", output_folder, pr_sens, out_xsize, out_ysize, new_gt)
        write_image(f"ps_{c}_{l}", output_folder, ps_sens, out_xsize, out_ysize, new_gt)
        write_image(f"tas_{c}_{l}", output_folder, tas_sens, out_xsize, out_ysize, new_gt)
        write_image(f"rsds_{c}_{l}", output_folder, rsds_sens, out_xsize, out_ysize, new_gt)
        write_image(f"rlds_{c}_{l}", output_folder, rlds_sens, out_xsize, out_ysize, new_gt)
        write_image(f"tdew_{c}_{l}", output_folder, tdew_sens, out_xsize, out_ysize, new_gt)
        write_image(f"RH_{c}_{l}", output_folder, rh_sens, out_xsize, out_ysize, new_gt)
        write_image(f"wind_{c}_{l}", output_folder, wind_sens, out_xsize, out_ysize, new_gt)
        write_image(f"Q_{c}_{l}", output_folder, Q_sens, out_xsize, out_ysize, new_gt)

if __name__ == "__main__":
    blocky = 40
    blockys = list(range(blocky))
    # for l in blockys:
    #l=10
    #main(l)

    with ProcessPoolExecutor(max_workers=40) as pool:
    	pool.map(main, blockys)