;************************************************
load "$NCARG_NCARG/nclscripts/csm/gsn_code.ncl"
load "$NCARG_NCARG/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_NCARG/nclscripts/csm/contributed.ncl"
;************************************************
; Regrid 2d variables to 1x1.25 resolution
;************************************************
begin
;************************************************
; Variable and file handling
;************************************************

; Load all file paths to be regridded
; path = asciiread("/home/bgb2/itcz_phys/regrid/regridpaths_2d_fullrad.txt",-1,"string")
path = asciiread("/home/bgb2/itcz_phys/regrid/regridpaths_2d_fullrad_ACCESS.txt",-1,"string")
npath = dimsizes(path)

; Model names
modelname = (/"ACCESS-ESM1-5","BCC-CSM2-MR","CESM2","CMCC-ESM2",\
  "CNRM-ESM2-1","EC-Earth3-CC","GFDL-ESM4","GISS-E2-1-G","IPSL-CM6A-LR",\
  "MIROC-ES2L","MPI-ESM1-2-LR","NorESM2-LM","UKESM1-0-LL"/)
nmodel = dimsizes(modelname)

; Define start time indices of 40-year period relative to branch time
is = 1200

; For naming the output file
timeperiod = asciiread("/home/bgb2/itcz_phys/regrid/newtimesco2.txt",-1,"string")

; Define new grid data using the GFDL-ESM4 grid (1x1.25)
fnewlat = addfile("/tiger/scratch/gpfs/GEOCLIM/bgb2/CMIP/GFDL-ESM4/areacella_fx_GFDL-ESM4_piControl_r1i1p1f1_gr1.nc","r")
; latf = fnewlat->lat(::-1)
lat = fnewlat->lat
lon = fnewlat->lon
opt = False

; New grid resolution
nlat = dimsizes(lat)
nlon = dimsizes(lon)

; Length of time period
ntime = 40*12

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
  nlat_o = dimsizes(camfile->lat)
  nlon_o = dimsizes(camfile->lon)

  ; var = new((/ntime,nlat_o,nlon_o/),"float")    ; Define variable to store data with original grid
  ; var_new = new((/ntime,nlat,nlon/), "float")   ; Define variable to store data with new grid

  ; var = camfile->$attrs(0)$(is:is+ntime-1,:,:)
  var = camfile->$attrs(0)$(time | is:is+ntime-1, lat | :, lon | :)
  var_new = area_conserve_remap_Wrap(var&lon,var&lat,var,lon,lat,opt)
  var_new = var_new(:,::-1,:)

  ; Write out variable at new resolution
  print("output file opened " + attrs(0)+" "+attrs(2)+" "+attrs(3))
  dirup = "/tiger/scratch/gpfs/GEOCLIM/bgb2/CMIP1x1.25/"+attrs(2)+"/"
  filo = dirup+str_join(attrs,"_")+"_"+timeperiod(ixmodel)+".nc"
  system("/bin/rm -f " + filo);
  fout = addfile(filo, "c");
  fout->lat = lat
  fout->lon = lon
  fout->$attrs(0)$ = var_new
  print(filo)

  delete(nlat_o)
  delete(nlon_o)
  delete(var)
  delete(var_new)

end do
end
