import os
import numpy as np
import xarray as xr
import pickle as pkl
   

def get_northward_heat_transport_from_flux(flux):
    '''
    Computes the integrated northward heat transport at
    each latitude from a heat flux following the methodology
    of Donohoe et al. (2020).

    HT(\Theta) = -2 \pi a^2 \int_\Theta^90 {\cos(\Theta) FLUX(\Theta) d\Theta}

    Parameters
    ----------
    flux : xarray.DataArray
        Heat flux with lat, lon, and time dimensions.

    Returns
    -------
    ht : xarray.DataArray
        Heat transport with lat and time dimensions.

    Author: Ben Buchovecky
    (Adapted from Matlab script by Lily Hahn)
    '''
    re = 6371220            # radius of Earth
    lat = flux.lat
    nlat = lat.size
    ntime = flux.time.size

    # Zonal mean flux
    flux_lat = flux.mean(dim='lon').squeeze()

    # Global mean from zonal mean computed using latitude-weighted average
    coslat_weights = np.cos(lat * np.pi / 180)
    flux_globmean = (flux_lat * coslat_weights).sum(dim='lat') / coslat_weights.sum()

    # Difference from global mean, ensures zero transport at poles (?)
    flux_latdiff = flux_lat.copy(deep=True)
    for im in range(ntime):
        flux_latdiff[im] = flux_lat[im] - flux_globmean[im].values

    ht = flux_lat.copy(deep=True)
    ht.name = 'northward heat transport'

    # Integrate heat flux anomaly from theta to 90N, get the transport
    for ilat in range(1, nlat-1):
        lat_tmp = lat[ilat:].values
        flux_tmp = flux_latdiff[:, ilat:].values
        # flux_tmp = flux_tmp[~np.isnan(flux_tmp)]  # remove NaN values -> shouldn't be any NaNs
        ht[:, ilat] = -2 * np.pi * (re**2) * np.trapz(flux_tmp, np.sin(np.deg2rad(lat_tmp)))

    return ht


def get_heat_transports_seasonal(modelname,
                                 experiment,
                                 write=False,
                                 write_path='/tiger/scratch/gpfs/GEOCLIM/bgb2/CMIPnht/',
                                 ):
    '''
    Computes the meridional heat transport, oceanic heat transport,
    atmospheric heat transport, moist atmospheric heat transport,
    and dry atmospheric heat transport accounting for atmospheric
    heat storage and following the methodology of Donohoe et al.
    (2020).

    Parameters
    ----------
    modelname : str
        Model name of interest following CMIP6 naming convention.
    experiment : str
        Experiment name(s) following CMIP6 naming convention.
    write : bool, optional
        If True, writes all heat transports to NetCDF files.
    write_path : str, optional
        The path to the directory where heat transport files will be saved.

    Returns
    -------
    ht : dict
        Dictionary with heat transport name keys and the corresponding
        DataArrays as items.

    Author: Ben Buchovecky
    (Adapted from Matlab script by Lily Hahn)
    '''
   
    ## Helper functions
    def gettimeslice(m, e):
        if e == 'piControl':
            st = period_l40_start_year_in_pi[m].replace('-','')
            et = period_l40_end_year_in_pi[m].replace('-','')
            time = st+'-'+et
        elif e[:7] == '1pctCO2':
            st = period_l40_start_year_in_co2[m].replace('-','')
            et = period_l40_end_year_in_co2[m].replace('-','')
            time = st+'-'+et
        else:
            time = ''
        return time
    
    def getcmippath(v, m, e):
        time = gettimeslice(m, e)
        return cmipgriddir+m+'/'+v+'_'+table_id[v]+'_'+m+'_'+e+'_'+variant_id[m]+'_1x1.25_'+time+'.nc'
    
    def getstoragepath(v, m, e):
        time = gettimeslice(m, e)
        return cmipnhtdir+m+'/'+v+'_Amon_'+m+'_'+e+'_'+variant_id[m]+'_1x1.25_'+time+'.nc'
    
    ## Directories
    cmipdir = '/tiger/scratch/gpfs/GEOCLIM/bgb2/CMIP/'
    cmipmergedir = '/tiger/scratch/gpfs/GEOCLIM/bgb2/CMIPmerge/'
    cmipgriddir = '/tiger/scratch/gpfs/GEOCLIM/bgb2/CMIP1x1.25/'
    cmipnhtdir = '/tiger/scratch/gpfs/GEOCLIM/bgb2/CMIPnht/'
    
    ## Load dictionaries
    # Get the last 40 years of the 1% CO2 runs and the corresponding time period in piControl
    with open('../pkl_files/period_l40_slice_in_pi.pkl', 'rb') as file:
        period_l40_slice_in_pi = pkl.load(file)
    with open('../pkl_files/period_l40_start_year_in_pi.pkl', 'rb') as file:
        period_l40_start_year_in_pi = pkl.load(file)
    with open('../pkl_files/period_l40_end_year_in_pi.pkl', 'rb') as file:
        period_l40_end_year_in_pi = pkl.load(file)
    with open('../pkl_files/start_year_in_pi.pkl', 'rb') as file:
        start_year_in_pi = pkl.load(file)

    with open('../pkl_files/period_l40_slice_in_co2.pkl', 'rb') as file:
        period_l40_slice_in_co2 = pkl.load(file)
    with open('../pkl_files/period_l40_start_year_in_co2.pkl', 'rb') as file:
        period_l40_start_year_in_co2 = pkl.load(file)
    with open('../pkl_files/period_l40_end_year_in_co2.pkl', 'rb') as file:
        period_l40_end_year_in_co2 = pkl.load(file) 
    with open('../pkl_files/start_year_in_co2.pkl', 'rb') as file:
        start_year_in_co2 = pkl.load(file)
    
    # Get each model's variant_id
    with open('../pkl_files/variant_id.pkl', 'rb') as file:
        variant_id = pkl.load(file)

    # Get each model's table_id
    with open('../pkl_files/table_id.pkl', 'rb') as file:
        table_id = pkl.load(file)

    # Get each model's grid_label
    with open('../pkl_files/grid_label.pkl', 'rb') as file:
        grid_label = pkl.load(file)

    if experiment == 'piControl':
        t = period_l40_slice_in_pi[modelname]
    elif experiment[:7] == '1pctCO2':
        t = period_l40_slice_in_co2[modelname]
    else:
        t = slice('', '')

    # Load data - rsdt, rsut, rlut
    rsdt = xr.open_dataset(getcmippath('rsdt', modelname, experiment))['rsdt']
    rsut = xr.open_dataset(getcmippath('rsut', modelname, experiment))['rsut']
    rlut = xr.open_dataset(getcmippath('rlut', modelname, experiment))['rlut']

    # Find meridional heat transport (MHT) by integrating net toa flux
    toaflux = rsdt - rsut - rlut
    mht = get_northward_heat_transport_from_flux(toaflux)
    mht.name = 'mht'
    mht.attrs['long_name'] = 'Meridional Heat Transport'
    mht.attrs['units'] = 'W'
    # del toaflux, rsdt, rsut, rlut
    print('### (1/5) done with mht ###')

    # Load data - rsds, rsus, rlds, rlus, hfls, hfss, prsn
    rsds = xr.open_dataset(getcmippath('rsds', modelname, experiment))['rsds']
    rsus = xr.open_dataset(getcmippath('rsus', modelname, experiment))['rsus']
    rlds = xr.open_dataset(getcmippath('rlds', modelname, experiment))['rlds']
    rlus = xr.open_dataset(getcmippath('rlus', modelname, experiment))['rlus']
    hfls = xr.open_dataset(getcmippath('hfls', modelname, experiment))['hfls']
    hfss = xr.open_dataset(getcmippath('hfss', modelname, experiment))['hfss']
    prsn = xr.open_dataset(getcmippath('prsn', modelname, experiment))['prsn']
    snowflux = prsn * 334000  # convert [kg m-2 s-1] to [W m-2]

    # Find ocean heat transprt (OHT) by integrating net surface heat flux
    shf = rsds - rsus + rlds - rlus - hfls - hfss - snowflux
    oht = get_northward_heat_transport_from_flux(shf)
    oht.name = 'oht'
    oht.attrs['long_name'] = 'Oceanic Heat Transport'
    oht.attrs['units'] = 'W'
    # del shf, rsds, rsus, rlds, rlus, hfss, prsn, snowflux
    print('### (2/5) done with oht ###')

    # Load storage terms - storage, storagemoist
    storage = xr.open_dataset(getstoragepath('storage', modelname, experiment))['storage']
    storagemoist = xr.open_dataset(getstoragepath('storagemoist', modelname, experiment))['storagemoist']

    # Ensure correct time periods
    storage = storage.sel(time=t)
    storagemoist = storagemoist.sel(time=t)

    # Integrate storage terms poleward
    storage = get_northward_heat_transport_from_flux(storage)
    storage.name = 'storage'
    storage.attrs['long_name'] = 'Atmospheric Storage'
    storage.attrs['units'] = 'W'

    storagemoist = get_northward_heat_transport_from_flux(storagemoist)
    storagemoist.name = 'storagemoist'
    storagemoist.attrs['long_name'] = 'Atmospheric Moisture Storage'
    storagemoist.attrs['units'] = 'W'

    # Find atmospere heat transport (AHT)
    aht = mht - oht - storage
    aht.name = 'aht'
    aht.attrs['long_name'] = 'Seasonal Atmospheric Heat Transport'
    aht.attrs['units'] = 'W'
    print('### (3/5) done with aht ###')

    # Load data - pr
    pr = xr.open_dataset(getcmippath('pr', modelname, experiment))['pr']
    
    # Moist AHT
    pr = pr * 2.5e6  # convert [kg m-2 s-1] to [W m-2]
    lhflux = hfls - pr
    ahtmoist = get_northward_heat_transport_from_flux(lhflux)
    ahtmoist = ahtmoist - storagemoist
    ahtmoist.name = 'ahtmoist'
    ahtmoist.attrs['long_name'] = 'Seasonal Moist Atmospheric Heat Transport'
    ahtmoist.attrs['units'] = 'W'
    # del hfls, pr
    print('### (4/5) done with ahtmoist ###')

    # Dry AHT
    ahtdry = aht - ahtmoist
    ahtdry.name = 'ahtdry'
    ahtdry.attrs['long_name'] = 'Seasonal Dry Atmospheric Heat Transport'
    ahtdry.attrs['units'] = 'W'
    print('### (5/5) done with ahtdry ###')

    ht = {'mht': mht, 'aht': aht, 'oht': oht, 'ahtmoist': ahtmoist, 'ahtdry': ahtdry,
          'storage': storage, 'storagemoist': storagemoist}

    if write:
        for key in ht.keys():
            fname = key+'_Amon_'+modelname+'_'+experiment+'_'+variant_id[modelname]+'_zonal_'+gettimeslice(modelname,experiment)+'.nc'
            ht[key].to_netcdf(write_path+modelname+'/'+fname)
            print(write_path+modelname+'/'+fname)

    return ht