;************************************************
load "$NCARG_NCARG/nclscripts/csm/gsn_code.ncl"
load "$NCARG_NCARG/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_NCARG/nclscripts/csm/contributed.ncl"
;************************************************
; Vertical interpolation to pressure grid on 3-D variables
; For 2d variables, use 2d_interp.ncl
;************************************************
begin
;************************************************
; Variable and file handling
;************************************************

path = asciiread("/home/bgb2/itcz_phys/regrid/regridpaths_3d_fullrad.txt",-1,"string")
npath = dimsizes(path)

; Model names (order matters for is_models!)
modelname = (/"ACCESS-ESM1-5","BCC-CSM2-MR","CESM2","CMCC-ESM2",\
  "CNRM-ESM2-1","EC-Earth3-CC","GFDL-ESM4","GISS-E2-1-G","IPSL-CM6A-LR",\
  "MIROC-ES2L","MPI-ESM1-2-LR","NorESM2-LM","UKESM1-0-LL"/)
nmodel = dimsizes(modelname)

; Define start time indices of 40-year period relative to branch time
is = 1200

; For naming the output file
timeperiod = asciiread("/home/bgb2/itcz_phys/regrid/newtimesco2.txt",13,"string")

; Define new grid
gfdl = addfile("/tiger/scratch/gpfs/GEOCLIM/bgb2/CMIP/GFDL-ESM4/hus_Amon_GFDL-ESM4_1pctCO2_r1i1p1f1_gr1_000101-010012.nc", "r")
new_plev = gfdl->plev
new_plev!0 = "plev"                      ; variable/dimension name
new_plev&plev = new_plev                 ; create coordinate variable
new_plev@long_name = "pressure"          ; attach some attributes
new_plev@standard_name = "air_pressure"
new_plev@units = "hPa"
latf = gfdl->lat(::-1)
lat = gfdl->lat
lon = gfdl->lon
opt = False

; New grid resolution
nlat = dimsizes(lat)
nlon = dimsizes(lon)
nlev = dimsizes(new_plev)

; Length of time period
ntime = 40*12  ; 40 years

; Type of interpolation: 1=linear 2=log 3=loglog
interp = 1

do ii = 3, npath-1
  ; Get attributes from file path
  tmpattrs = str_split(path(ii),"/")
  attrs = str_split(tmpattrs(7),"_")
  attrs(5) = "1x1.25"

  ; Find the index corresponding to the model
  do mm = 0, nmodel-1
    if attrs(2).eq.modelname(mm)
      ixmodel = mm
      break
    end if
  end do

  ; Interpolate
  camfile = addfile(path(ii), "r")
  old_plev = camfile->plev
  nlat_o = dimsizes(camfile->lat)
  nlon_o = dimsizes(camfile->lon)
  nlev_o = dimsizes(camfile->plev)

  ; Open file and extract variable within appropriate time bounds
  var = camfile->$attrs(0)$(time | is:is+ntime-1, plev | :, lat | :, lon | :)

  ; Interpolate onto 2d grid
  var_2dinterp = area_conserve_remap_Wrap(var&lon, var&lat, var, lon, lat, opt)
  var_2dinterp(:,:,:,:) = var_2dinterp(:,:,::-1,:)

  ; Interpolate onto new pressure levels
  var_3dinterp = int2p_n_Wrap(old_plev, var_2dinterp, new_plev, interp, 1)
  var_3dinterp(:,:,:,:) = var_3dinterp(:,:,::-1,:)

  ; Write out variable at new resolution
  print("output file opened " + attrs(0)+" "+attrs(2)+" "+attrs(3))
  dirup = "/tiger/scratch/gpfs/GEOCLIM/bgb2/CMIP1x1.25/"+attrs(2)+"/"
  filo = dirup+str_join(attrs,"_")+"_"+timeperiod(ixmodel)+".nc"
  system("/bin/rm -f " + filo);
  fout = addfile(filo, "c");
  fout->lat = latf
  fout->lon = lon
  fout->plev = new_plev
  fout->$attrs(0)$ = var_3dinterp
  print(filo)

  delete(filo)
  delete(nlat_o)
  delete(nlon_o)
  delete(camfile)
  delete(var)
  delete(var_2dinterp)
  delete(var_3dinterp)
end do
end
