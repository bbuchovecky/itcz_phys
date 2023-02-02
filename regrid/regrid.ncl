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
path = asciiread("/glade/u/home/bbuchovecky/itcz_phys/regrid/regrid_paths.txt",-1,"string")
n_path = dimsizes(path)

; Model names (order matters for is_models!)
model_name = (/"ACCESS-ESM1-5","BCC-CSM2-MR","CanESM5","CESM2","CMCC-ESM2",\
  "CNRM-ESM2-1","GFDL-ESM4","GISS-E2-1-G","IPSL-CM6A-LR","MIROC-ES2L",\
  "MPI-ESM1-2-LR","NorESM2-LM","UKESM1-0-LL"/)
n_model = dimsizes(model_name)

; Define start time indices of 40-year period relative to branch time
is_models = asciiread("/glade/u/home/bbuchovecky/itcz_phys/regrid/is_models.txt",(/13,2/),"float")

; For naming the output file
time_period = asciiread("/glade/u/home/bbuchovecky/itcz_phys/regrid/new_tperiods.txt",(/13,2/),"string")

; Define new grid data using the GFDL-ESM4 grid (1x1.25)
fnewlat = addfile("/glade/scratch/bbuchovecky/cmip/GFDL-ESM4/areacella_fx_GFDL-ESM4_piControl_r1i1p1f1_gr1.nc","r")
; latf = fnewlat->lat(::-1)
lat = fnewlat->lat
lon = fnewlat->lon
opt = False

; New grid resolution
nlat = dimsizes(lat)
nlon = dimsizes(lon)

; Length of time period
n_time = 40*12

do ii = 4, n_path-1
  ; Get attributes from file path
  tmp_attrs = str_split(path(ii),"/")
  attrs = str_split(tmp_attrs(5),"_")
  attrs(5) = "1x1.25"

  ; Find the index corresponding to the model
  do mm = 0, n_model-1
    if attrs(2).eq.model_name(mm)
      ix_model = mm
      break
    end if
  end do

  ; Get start index, checking if piControl or 1pctCO2 experiment
  if attrs(3).eq."piControl"
    is = floattointeger(is_models(ix_model,0))
    attrs(6) = time_period(ix_model,0)+".nc"
  else
    is = floattointeger(is_models(ix_model,1))
    attrs(6) = time_period(ix_model,1)+".nc"
  end if

  ; Interpolate
  camfile = addfile(path(ii), "r")
  nlat_o = dimsizes(camfile->lat)
  nlon_o = dimsizes(camfile->lon)
  
  var = new((/n_time,nlat_o,nlon_o/),"float")    ; Define variable to store data with original grid
  var_new = new((/n_time,nlat,nlon/), "float")   ; Define variable to store data with new grid
  
  var = camfile->$attrs(0)$(is:is+n_time-1,:,:)
  var_new = area_conserve_remap_Wrap(var&lon,var&lat,var,lon,lat,opt)
  var_new = var_new(:,::-1,:)
  ; printVarSummary(var_new)

  ; Write out variable at new resolution
  print("output file opened " + attrs(0)+" "+attrs(1)+" "+attrs(2)+" "+attrs(3)+" "+attrs(4)+" "+attrs(5)+" "+attrs(6))
  dirup = "/glade/scratch/bbuchovecky/cmip_1x1.25/"+attrs(2)+"/"
  filo = dirup+str_join(attrs,"_")
  system("/bin/rm -f " + filo);
  fout = addfile(filo, "c");
  fout->lat = lat
  fout->lon = lon
  fout->$attrs(0)$ = var_new
  ; print(filo)
  
  delete(nlat_o)
  delete(nlon_o)
  delete(var)
  delete(var_new)

end do
end