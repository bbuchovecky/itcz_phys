import os
import intake
import numpy as np
import xarray as xr

def getcmippath(v,
                m,
                merge=True):
    '''
    Returns the path to the CMIP6 file on /tiger/scratch
    
    Parameters:
    -----------
    v : str
        Variable name.
    m : str
        Model name.
    
    Returns:
    --------
    path : str
        Path to the CMIP6 file.
    '''
    if merge:
        return cmipmergedir+m+'/'+v+'_'+table_id[v]+'_'+m+'_'+c+'_'+variant_id[m]+'_'+grid_label[m]+'.nc'
    if not merge:
        return cmipdir+m+'/'+v+'_'+table_id[v]+'_'+m+'_'+c+'_'+variant_id[m]+'_'+grid_label[m]+'_*.nc'
    

def get_northward_heat_transport_from_flux(flux,
                                           single_member=True,
                                           mmm=True):
    '''
    Computes the integrated northward heat transport at
    each latitude from a heat flux following the methodology
    of Donohoe et al. (2020).

    HT(\Theta) = -2 \pi a^2 \int_\Theta^90 {\cos(\Theta) FLUX(\Theta) d\Theta}

    Parameters
    ----------
    flux : xarray.DataArray
        Heat flux with lat, lon, and time dimensions.
    single_member : bool, optional
        If True, calculates the heat transport for the first member only.
    mmm : bool, optional
        If True, takes the multi-member mean heat transport.

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

    # Some logic to check if there are more than one member
    # ** Multi-member heat transport NOT FULLY IMPLEMENTED **
    try:
        print('number of members:', flux['member_id'].size)
        if single_member and flux['member_id'].size > 1:
            flux = flux.isel(member_id=0).squeeze()
            mmm = False
    except KeyError:
        single_member = True
        mmm = False

    # Zonal mean flux
    flux_lat = flux.mean(dim='lon').squeeze()

    # Global mean from zonal mean computed using latitude-weighted average
    coslat_weights = np.cos(lat * np.pi / 180)
    flux_globmean = (flux_lat * coslat_weights).sum(dim='lat') / coslat_weights.sum()

    # Difference from global mean, ensures zero transport at poles (?)
    flux_latdiff = flux_lat.copy(deep=True)
    if not single_member:
        for im in range(ntime):
            flux_latdiff[:, im] = flux_lat[:, im] - flux_globmean[:, im]
    else:
        for im in range(ntime):
            flux_latdiff[im] = flux_lat[im] - flux_globmean[im].values

    ht = flux_lat.copy(deep=True)
    ht.name = 'northward heat transport'

    # Integrate heat flux anomaly from theta to 90N, get the transport
    # Take ensemble mean if >1 members
    if not single_member:
        for ilat in range(1, nlat-1):
            lat_tmp = lat[ilat:].values
            flux_tmp = flux_latdiff[:, :, ilat:].values
            # flux_tmp = flux_tmp[~np.isnan(flux_tmp)]  # remove NaN values -> shouldn't be any NaNs
            ht[:, :, ilat] = -2 * np.pi * (re**2) * np.trapz(flux_tmp, np.sin(np.deg2rad(lat_tmp)))
        if mmm:
            ht = ht.mean(dim='member_id').squeeze()
    else:
        for ilat in range(1, nlat-1):
            lat_tmp = lat[ilat:].values
            flux_tmp = flux_latdiff[:, ilat:].values
            # flux_tmp = flux_tmp[~np.isnan(flux_tmp)]  # remove NaN values -> shouldn't be any NaNs
            ht[:, ilat] = -2 * np.pi * (re**2) * np.trapz(flux_tmp, np.sin(np.deg2rad(lat_tmp)))

    return ht


def get_heat_transports_seasonal_multimodel(modelname,
                                            experiment,
                                            grid_label,
                                            col=None,
                                            write=False,
                                            write_path='/glade/scratch/bbuchovecky/heat_transport/',
                                            **kwargs):
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
    grid_label : str
        Desired grid types following CMIP6 naming convention.
    col : intake_esm.core.esm_datastore, optional
        Created with intake-esm from a json catalog file.
        On NCAR servers, the catalog file is located at
        /glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json
        Using the col parameter will speed up data retrieval.
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
    # Decide whether to use multiple members, corresponds to
    # get_northward_heat_transport_from_flux()
    # Dimensions will likely not match if single_member=False and mmm=False
    # ** Multi-member heat transport NOT FULLY IMPLEMENTED **
    single_member = True
    mmm = False

    # Time periods for last 40 years of 1pctCO2 runs that match storage terms
    sel_time_periods = {'CNRM-ESM2-1': {'piControl': '196001-199912', '1pctCO2': '196001-199912',
                                    '1pctCO2-rad': '196001-199912', '1pctCO2-bgc': '196001-199912'},

                        'BCC-CSM2-MR': {'piControl': '196001-199912', '1pctCO2': '196001-199912',
                                        '1pctCO2-rad': '196001-199912', '1pctCO2-bgc': '196001-199912'},

                        'CanESM5': {'piControl': '531101-535012', '1pctCO2': '196001-199912',
                                    '1pctCO2-rad': '196001-199912', '1pctCO2-bgc': '196001-199912'},

                        'CESM2': {'piControl': '061101-065012', '1pctCO2': '011101-015012',
                                  '1pctCO2-rad': '011101-015012', '1pctCO2-bgc': '011101-015012'},

                        'UKESM1-0-LL': {'piControl': '207001-210912', '1pctCO2': '196001-199912',
                                        '1pctCO2-rad': '196001-199912', '1pctCO2-bgc': '196001-199912'},

                        'GISS-E2-1-G': {'piControl': '426001-429912', '1pctCO2': '196001-199912',
                                        '1pctCO2-rad': '196001-199912', '1pctCO2-bgc': '196001-199912'},

                        'GFDL-ESM4': {'piControl': '021101-025012', '1pctCO2': '011101-015012',
                                      '1pctCO2-rad': '011101-015012', '1pctCO2-bgc': '011101-015012'},

                        'IPSL-CM6A-LR': {'piControl': '198001-201912', '1pctCO2': '196001-199912',
                                         '1pctCO2-rad': '196001-199912', '1pctCO2-bgc': '196001-199912'},

                        'MIROC-ES2L': {'piControl': '196001-199912', '1pctCO2': '196001-199912',
                                       '1pctCO2-rad': '196001-199912', '1pctCO2-bgc': '196001-199912'},

                        'MRI-ESM2-0': {'piControl': '196001-199912', '1pctCO2': '196001-199912',
                                       '1pctCO2-rad': '196001-199912', '1pctCO2-bgc': '196001-199912'},

                        'MPI-ESM1-2-LR': {'piControl': '196001-199912', '1pctCO2': '',
                                          '1pctCO2-rad': '', '1pctCO2-bgc': ''},

                        'ACCESS-ESM1-5': {'piControl': '021101-025012', '1pctCO2': '021101-025012',
                                          '1pctCO2-rad': '021101-025012', '1pctCO2-bgc': '021101-025012'}}

    times = sel_time_periods[modelname][experiment].split('-')
    if len(times) == 2:
        t = slice(times[0][:4]+'-'+times[0][4:], times[1][:4]+'-'+times[1][4:])
    else:
        t = slice('', '')

    # Load data
    rsdt = get_cmip_data(modelname, 'rsdt', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rsut = get_cmip_data(modelname, 'rsut', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rlut = get_cmip_data(modelname, 'rlut', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()

    # Find meridional heat transport (mht) by integrating net toa flux
    toaflux = rsdt - rsut - rlut
    mht = get_northward_heat_transport_from_flux(toaflux, single_member=single_member, mmm=mmm)
    mht.name = 'mht'
    mht.attrs['long_name'] = 'Meridional Heat Transport'
    mht.attrs['units'] = 'W'
    # del toaflux, rsdt, rsut, rlut
    print('### (1/5) done with mht ###')

    # Load data
    rsds = get_cmip_data(modelname, 'rsds', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rsus = get_cmip_data(modelname, 'rsus', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rlds = get_cmip_data(modelname, 'rlds', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rlus = get_cmip_data(modelname, 'rlus', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    hfls = get_cmip_data(modelname, 'hfls', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    hfss = get_cmip_data(modelname, 'hfss', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    prsn = get_cmip_data(modelname, 'prsn', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    snowflux = prsn * 334000  # convert [kg m-2 s-1] to [W m-2]

    # Find ocean heat transprt (oht) by integrating net surface heat flux
    shf = rsds - rsus + rlds - rlus - hfls - hfss - snowflux
    oht = get_northward_heat_transport_from_flux(shf, single_member=single_member, mmm=mmm)
    oht.name = 'oht'
    oht.attrs['long_name'] = 'Oceanic Heat Transport'
    oht.attrs['units'] = 'W'
    # del shf, rsds, rsus, rlds, rlus, hfss, prsn, snowflux
    print('### (2/5) done with oht ###')

    # Load storage terms
    storage = xr.open_dataset('/glade/scratch/bbuchovecky/heat_transport/'+modelname+'/storage_gc_'+modelname+'_'+experiment+'.nc')['storage']
    storagemoist = xr.open_dataset('/glade/scratch/bbuchovecky/heat_transport/'+modelname+'/storagemoist_gc_'+modelname+'_'+experiment+'.nc')['storagemoist']

    # Ensure correct time periods
    storage = storage.sel(time=t)
    storagemoist = storagemoist.sel(time=t)

    # Integrate storage terms poleward
    storage = get_northward_heat_transport_from_flux(storage, single_member=single_member, mmm=mmm)
    storage.name = 'storage'
    storage.attrs['long_name'] = 'Atmospheric Storage'
    storage.attrs['units'] = 'W'

    storagemoist = get_northward_heat_transport_from_flux(storagemoist, single_member=single_member, mmm=mmm)
    storagemoist.name = 'storagemoist'
    storagemoist.attrs['long_name'] = 'Atmospheric Moisture Storage'
    storagemoist.attrs['units'] = 'W'

    # Find atmospere heat transport (aht)
    aht = mht - oht - storage
    aht.name = 'aht'
    aht.attrs['long_name'] = 'Seasonal Atmospheric Heat Transport'
    aht.attrs['units'] = 'W'
    print('### (3/5) done with aht ###')

    # Load data
    pr = get_cmip_data(modelname, 'pr', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()

    # moist aht
    pr = pr * 2.5e6  # convert [kg m-2 s-1] to [W m-2]
    lhflux = hfls - pr
    ahtmoist = get_northward_heat_transport_from_flux(lhflux, single_member=single_member, mmm=mmm)
    ahtmoist = ahtmoist - storagemoist
    ahtmoist.name = 'ahtmoist'
    ahtmoist.attrs['long_name'] = 'Seasonal Moist Atmospheric Heat Transport'
    ahtmoist.attrs['units'] = 'W'
    # del hfls, pr
    print('### (4/5) done with ahtmoist ###')

    # dry aht
    ahtdry = aht - ahtmoist
    ahtdry.name = 'ahtdry'
    ahtdry.attrs['long_name'] = 'Seasonal Dry Atmospheric Heat Transport'
    ahtdry.attrs['units'] = 'W'
    print('### (5/5) done with ahtdry ###')

    ht = {'mht': mht, 'aht': aht, 'oht': oht, 'ahtmoist': ahtmoist, 'ahtdry': ahtdry,
          'storage': storage, 'storagemoist': storagemoist}

    if write:
        for key in ht.keys():
            if mmm:
                fname = key+'_'+modelname+'_mme_'+experiment+'.nc'
            else:
                fname = key+'_'+modelname+'_'+experiment+'.nc'
            ht[key].to_netcdf(write_path+modelname+'/'+fname)
            print(write_path+modelname+'/'+fname)

    return ht

#######################################
######## Depreciated functions ########
#######################################


def get_heat_transports(modelname,
                        experiment,
                        grid_label,
                        mmm=True,
                        col=None,
                        write=False,
                        write_path='/glade/scratch/bbuchovecky/heat_transport/',
                        **kwargs):
    '''
    Computes the meridional heat transport, oceanic heat transport,
    atmospheric heat transport, moist atmospheric heat transport,
    and dry atmospheric heat transport following the methodology of
    Donohoe et al. (2020).

    Parameters
    ----------
    modelname : str
        Model name of interest following CMIP6 naming convention.
    experiment : str
        Experiment name(s) following CMIP6 naming convention.
    grid_label : str
        Desired grid types following CMIP6 naming convention.
    col : intake_esm.core.esm_datastore, optional
        Created with intake-esm from a json catalog file.
        On NCAR servers, the catalog file is located at
        /glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json
        Using the col parameter will speed up data retrieval.
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
    # Decide whether to use multiple members, corresponds to
    # get_northward_heat_transport_from_flux()
    # Dimensions will likely not match if single_member=False and mmm=False
    single_member = True
    mmm = False

    # Time periods to match storage terms
    # tperiods = {'piControl': slice('2070-01', '2109-12'),
    #             '1pctCO2': slice('1960-01', '1999-12'),
    #             '1pctCO2-rad': slice('1960-01', '1999-12'),
    #             '1pctCO2-bgc': slice('1960-01', '1999-12')}
    # t = tperiods[experiment]
    t = slice('0000-01', '9999-12')

    # Load data
    rsdt = get_cmip_data(modelname, 'rsdt', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rsut = get_cmip_data(modelname, 'rsut', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rlut = get_cmip_data(modelname, 'rlut', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()

    # Find meridional heat transport (mht) by integrating net toa flux
    toaflux = rsdt - rsut - rlut
    mht = get_northward_heat_transport_from_flux(toaflux, single_member=single_member, mmm=mmm)
    mht.name = 'mht'
    mht.attrs['long_name'] = 'Meridional Heat Transport'
    mht.attrs['units'] = 'W'
    del toaflux, rsdt, rsut, rlut
    print('### (1/5) done with mht ###')

    # Load data
    rsds = get_cmip_data(modelname, 'rsds', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rsus = get_cmip_data(modelname, 'rsus', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rlds = get_cmip_data(modelname, 'rlds', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rlus = get_cmip_data(modelname, 'rlus', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    hfls = get_cmip_data(modelname, 'hfls', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    hfss = get_cmip_data(modelname, 'hfss', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    prsn = get_cmip_data(modelname, 'prsn', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    snowflux = prsn * 334000  # convert [kg m-2 s-1] to [W m-2]

    # Find ocean heat transprt (oht) by integrating net surface heat flux
    shf = rsds - rsus + rlds - rlus - hfls - hfss - snowflux
    oht = get_northward_heat_transport_from_flux(shf, single_member=single_member, mmm=mmm)
    oht.name = 'oht'
    oht.attrs['long_name'] = 'Oceanic Heat Transport'
    oht.attrs['units'] = 'W'
    del shf, rsds, rsus, rlds, rlus, hfss, prsn, snowflux
    print('### (2/5) done with oht ###')

    # Find atmospere heat transport (aht)
    aht = mht - oht
    aht.name = 'aht'
    aht.attrs['long_name'] = 'Atmospheric Heat Transport'
    aht.attrs['units'] = 'W'
    print('### (3/5) done with aht ###')

    # Load data
    pr = get_cmip_data(modelname, 'pr', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()

    # Find moist aht
    pr = pr * 2.5e6  # convert [kg m-2 s-1] to [W m-2]
    lhflux = hfls - pr
    ahtmoist = get_northward_heat_transport_from_flux(lhflux, single_member=single_member, mmm=mmm)
    ahtmoist.name = 'ahtmoist'
    ahtmoist.attrs['long_name'] = 'Moist Atmospheric Heat Transport'
    ahtmoist.attrs['units'] = 'W'
    del hfls, pr
    print('### (4/5) done with ahtmoist ###')

    # Find dry aht
    ahtdry = aht - ahtmoist
    ahtdry.name = 'ahtdry'
    ahtdry.attrs['long_name'] = 'Dry Atmospheric Heat Transport'
    ahtdry.attrs['units'] = 'W'
    print('### (5/5) done with ahtdry ###')

    ht = {'mht': mht, 'aht': aht, 'oht': oht, 'ahtmoist': ahtmoist, 'ahtdry': ahtdry}

    if write:
        for key in ht.keys():
            if mmm:
                fname = key+'_'+modelname+'_mme_'+experiment+'.nc'
            else:
                fname = key+'_'+modelname+'_'+experiment+'.nc'
            ht[key].to_netcdf(write_path+fname)
            print(write_path+fname)

    return ht


def get_heat_transports_seasonal(modelname,
                                 experiment,
                                 grid_label,
                                 col=None,
                                 write=False,
                                 write_path='/glade/scratch/bbuchovecky/heat_transport/',
                                 **kwargs):
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
    grid_label : str
        Desired grid types following CMIP6 naming convention.
    col : intake_esm.core.esm_datastore, optional
        Created with intake-esm from a json catalog file.
        On NCAR servers, the catalog file is located at
        /glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json
        Using the col parameter will speed up data retrieval.
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
    # Decide whether to use multiple members, corresponds to
    # get_northward_heat_transport_from_flux()
    # Dimensions will likely not match if single_member=False and mmm=False
    single_member = True
    mmm = False

    # Time periods to match storage terms
    tperiods = {'piControl': slice('2070-01', '2109-12'),
                '1pctCO2': slice('1960-01', '1999-12'),
                '1pctCO2-rad': slice('1960-01', '1999-12'),
                '1pctCO2-bgc': slice('1960-01', '1999-12')}
    t = tperiods[experiment]

    # Load data
    rsdt = get_cmip_data(modelname, 'rsdt', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rsut = get_cmip_data(modelname, 'rsut', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rlut = get_cmip_data(modelname, 'rlut', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()

    # Find meridional heat transport (mht) by integrating net toa flux
    toaflux = rsdt - rsut - rlut
    mht = get_northward_heat_transport_from_flux(toaflux, single_member=single_member, mmm=mmm)
    mht.name = 'mht'
    mht.attrs['long_name'] = 'Meridional Heat Transport'
    mht.attrs['units'] = 'W'
    # del toaflux, rsdt, rsut, rlut
    print('### (1/5) done with mht ###')

    # Load data
    rsds = get_cmip_data(modelname, 'rsds', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rsus = get_cmip_data(modelname, 'rsus', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rlds = get_cmip_data(modelname, 'rlds', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    rlus = get_cmip_data(modelname, 'rlus', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    hfls = get_cmip_data(modelname, 'hfls', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    hfss = get_cmip_data(modelname, 'hfss', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    prsn = get_cmip_data(modelname, 'prsn', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()
    snowflux = prsn * 334000  # convert [kg m-2 s-1] to [W m-2]

    # Find ocean heat transprt (oht) by integrating net surface heat flux
    shf = rsds - rsus + rlds - rlus - hfls - hfss - snowflux
    oht = get_northward_heat_transport_from_flux(shf, single_member=single_member, mmm=mmm)
    oht.name = 'oht'
    oht.attrs['long_name'] = 'Oceanic Heat Transport'
    oht.attrs['units'] = 'W'
    # del shf, rsds, rsus, rlds, rlus, hfss, prsn, snowflux
    print('### (2/5) done with oht ###')

    # Load storage terms
    storage = xr.open_dataset('/glade/scratch/bbuchovecky/heat_transport/storage_'+modelname+'_'+experiment+'.nc')['storage']
    storagemoist = xr.open_dataset('/glade/scratch/bbuchovecky/heat_transport/storagemoist_'+modelname+'_'+experiment+'.nc')['storagemoist']

    # Ensure correct time periods
    storage = storage.sel(time=slice(t))
    storagemoist = storagemoist.sel(time=slice(t))

    # Integrate storage terms poleward
    storage = get_northward_heat_transport_from_flux(storage, single_member=single_member, mmm=mmm)
    storage.name = 'storage'
    storage.attrs['long_name'] = 'Atmospheric Storage'
    storage.attrs['units'] = 'W'

    storagemoist = get_northward_heat_transport_from_flux(storagemoist, single_member=single_member, mmm=mmm)
    storagemoist.name = 'storagemoist'
    storagemoist.attrs['long_name'] = 'Atmospheric Moisture Storage'
    storagemoist.attrs['units'] = 'W'

    # Find atmospere heat transport (aht)
    aht = mht - oht - storage
    aht.name = 'aht'
    aht.attrs['long_name'] = 'Seasonal Atmospheric Heat Transport'
    aht.attrs['units'] = 'W'
    print('### (3/5) done with aht ###')

    # Load data
    pr = get_cmip_data(modelname, 'pr', experiment, grid_label, col=col, **kwargs)[0][experiment].sel(time=t).load()

    # moist aht
    pr = pr * 2.5e6  # convert [kg m-2 s-1] to [W m-2]
    lhflux = hfls - pr
    ahtmoist = get_northward_heat_transport_from_flux(lhflux, single_member=single_member, mmm=mmm)
    ahtmoist = ahtmoist - storagemoist
    ahtmoist.name = 'ahtmoist'
    ahtmoist.attrs['long_name'] = 'Seasonal Moist Atmospheric Heat Transport'
    ahtmoist.attrs['units'] = 'W'
    # del hfls, pr
    print('### (4/5) done with ahtmoist ###')

    # dry aht
    ahtdry = aht - ahtmoist
    ahtdry.name = 'ahtdry'
    ahtdry.attrs['long_name'] = 'Seasonal Dry Atmospheric Heat Transport'
    ahtdry.attrs['units'] = 'W'
    print('### (5/5) done with ahtdry ###')

    ht = {'mht': mht, 'aht': aht, 'oht': oht, 'ahtmoist': ahtmoist, 'ahtdry': ahtdry,
          'storage': storage, 'storagemoist': storagemoist}

    if write:
        for key in ht.keys():
            if mmm:
                fname = key+'_seasonal_'+modelname+'_mme_'+experiment+'.nc'
            else:
                fname = key+'_seasonal_'+modelname+'_'+experiment+'.nc'
            ht[key].to_netcdf(write_path+fname)
            print(write_path+fname)

    return ht


# def get_cmip_data(modelname,
#                  variable,
#                  experiment,
#                  grid_label,
#                  table_id='Amon',
#                  col=None,
#                  useCatalog=True,
#                  primary_dir='/glade/scratch/bbuchovecky/cmip/',
#                  catalog_fname='/glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json'):
#    '''
#    CMIP6 data retrieval.
#    Relies on avail_models function.
#
#    Parameters
#    ----------
#    modelname : str
#        Model name of interest following CMIP6 naming convention.
#    variable : str
#        Variable name of interest following CMIP6 naming convention.
#    experiment : str
#        Experiment name(s) following CMIP6 naming convention.
#    grid_label : str
#        Grid type following CMIP6 naming convention.
#    table_id : str, optional
#        Table ID following CMIP6 naming convention.
#    col : intake_esm.core.esm_datastore, optional
#        Created with intake-esm from a json catalog file.
#        On NCAR servers, the catalog file is located at
#        /glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json
#        Using the col parameter will speed up data retrieval.
#    useCatalog : bool, optional
#        If True, will try to use the catalog. If False, will only
#        search files in primary_dir.
#    primary_dir : str, optional
#        The directory to search for files in.
#    catalog_fname : str, optional
#        The json catalog file to open.
#
#    Returns
#    -------
#    data_dict : dict
#        Dictionary with experiment name keys and the corresponding
#        DataArrays as items.
#    name_dict : dict
#        Dictionary with the model name, table ID, long variable name,
#        and variable units.
#
#    Author: Ben Buchovecky
#    '''
#
#    print('model        ', modelname)
#    print('variable     ', variable)
#    print('experiments  ', experiment)
#    print('-------------------')
#
#    if type(experiment) != list:
#        experiment = [experiment]
#
#    data_dict = {}
#    cat_runs = []
#
#    for runname in experiment:
#        try:
#            cat = get_cmip_files_dict(primary_dir+modelname+'/')
#            members = list(cat[variable][table_id][modelname][runname].keys())
#            tperiod = cat[variable][table_id][modelname][runname][members[0]][grid_label]
#            n_members = len(members)
#
#            if n_members > 1:
#                da_list = []
#                for t in tperiod:
#                    fname = primary_dir+modelname+'/'+variable+'_'+table_id+'_*'+modelname+'_*'+runname+'_*'+grid_label+'_'+t+'.nc'
#                    print(fname)
#                    da_list.append(xr.open_mfdataset(fname, concat_dim='member_id', combine='nested'))
#                ds = xr.concat(da_list, dim='time').assign_coords(member_id=members)
#            else:
#                fname = primary_dir+modelname+'/'+variable+'_'+table_id+'_*'+modelname+'_*'+runname+'_*'+grid_label+'_*.nc'
#                print(fname)
#                try:
#                    ds = xr.open_mfdataset(fname)
#                except TypeError:
#                    ds = xr.open_mfdataset(sorted(glob.glob(fname))[0])
#
#            data_dict[runname] = ds[variable].chunk('auto')
#            table_id = ds.attrs['table_id']
#
#        except (FileNotFoundError, KeyError):
#            print('one or more files not found in directory')
#            cat_runs.append(runname)
#
#    if isinstance(col, type(None)):
#        print('opening catalog')
#        col = intake.open_esm_datastore(catalog_fname)
#
#    model_list = avail_models(variable, experiment, grid_label, table_id, col=col)
#
#    if np.isin(modelname, model_list) and useCatalog:
#        print('model output is available in catalog')
#        cat = col.search(source_id=modelname,
#                         experiment_id=experiment,
#                         variable_id=variable,
#                         grid_label=grid_label,
#                         table_id=table_id)
#        dset_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True})
#
#        for runname in cat_runs:
#            ds_w_data = cat.df[(cat.df.experiment_id == runname)
#                       & (cat.df.variable_id == variable)]
#            models_w_data = ds_w_data.drop_duplicates(subset='source_id')
#            i = models_w_data.index[0]
#            datakey = (models_w_data.activity_id[i] + '.' +
#                       models_w_data.institution_id[i] + '.' +
#                       models_w_data.source_id[i] + '.' +
#                       runname + '.' +
#                       models_w_data.table_id[i] + '.' +
#                               models_w_data.grid_label[i])
#            data_dict[runname] = dset_dict[datakey][variable].chunk('auto').squeeze()
#
#        table_id = models_w_data.table_id[i]
#
#    ds = data_dict[list(data_dict.keys())[0]]
#    varname_long = ds.attrs['long_name']
#    varunits = ds.attrs['units']
#    name_dict = {'model_name': modelname, 'table_id': table_id,
#                 'long_name': varname_long, 'units': varunits}
#
#    return data_dict, name_dict