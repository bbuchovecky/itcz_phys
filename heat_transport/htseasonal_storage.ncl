;************************************************
load "$NCARG_NCARG/nclscripts/csm/gsn_code.ncl"
load "$NCARG_NCARG/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_NCARG/nclscripts/csm/contributed.ncl"
;************************************************
; Calculates seasonal AHT and AHTmoist (removing heat storage) so that
; AHTdry = AHT - AHTmoist
; OHT actually represents OHT + heat storage
;************************************************
begin
;************************************************
; Variable and file handling
;************************************************
dir = "/tiger/scratch/gpfs/GEOCLIM/bgb2/CMIP1x1.25/"
dirup = "/tiger/scratch/gpfs/GEOCLIM/bgb2/CMIPnht/"

casename = (/"1pctCO2", "1pctCO2-rad"/)
ncase = dimsizes(casename)

modelname = (/"ACCESS-ESM1-5","BCC-CSM2-MR","CESM2","CMCC-ESM2",\
  "CNRM-ESM2-1","EC-Earth3-CC","GFDL-ESM4","GISS-E2-1-G","IPSL-CM6A-LR",\
  "MIROC-ES2L","MPI-ESM1-2-LR","NorESM2-LM","UKESM1-0-LL"/)
nmodel = dimsizes(modelname)

variantid = (/"r1i1p1f1","r1i1p1f1","r1i1p1f1","r1i1p1f1","r1i1p1f2",\
  "r1i1p1f1","r1i1p1f1","r101i1p1f1","r1i1p1f1","r1i1p1f2","r1i1p1f1",\
  "r1i1p1f1","r1i1p1f2"/)

timeperiod = asciiread("/home/bgb2/itcz_phys/regrid/newtimesco2.txt",13,"string")

ntime = 40*12

; Constants for storage term
cp = 1004.0              ; J kg-1 K-1
Lv = 2500000.0           ; J kg-1
g = 9.81                 ; m s-2
dt = 365.*24.*3600./12.  ; s

; Iterate through models
do m = 0, nmodel-1
    print(modelname(m))

    ; Iterate through experiments
    do c = 0, ncase-1

      ; Generate paths for each variable
      ta_path = dir+modelname(m)+"/ta_Amon_"+modelname(m)+"_"+casename(c)+"_"+variantid(m)+"_1x1.25_"+timeperiod(m)+".nc"
      hus_path = dir+modelname(m)+"/hus_Amon_"+modelname(m)+"_"+casename(c)+"_"+variantid(m)+"_1x1.25_"+timeperiod(m)+".nc"
      ps_path = dir+modelname(m)+"/ps_Amon_"+modelname(m)+"_"+casename(c)+"_"+variantid(m)+"_1x1.25_"+timeperiod(m)+".nc"

      print(casename(c))
      print("Calculating storage terms")
      f = addfile(ta_path, "r")
      T = f->ta(time | :, plev | :, lat | :, lon | :)
      lat = f->lat
      lon = f->lon

      lev = f->plev  ; Pa
      if(lev@units.eq."hPa")
        lev = lev * 100
        lev@units = "Pa"
        print(lev@units)
      else
        print(lev@units)
      end if

      tlat = f->lat
      tlon = f->lon

      f = addfile(hus_path, "r")
      Q = f->hus(time | :, plev | :, lat | :, lon | :)
      Q = where(Q.lt.0.,Q@_FillValue,Q)  ; For GISS issues

      f = addfile(ps_path, "r")
      ps = f->ps(time | :, lat | :, lon | :)
      pslat = f->lat
      pslon = f->lon
      
      if(ps@units.eq."hPa")
        ps(time | :, lat | :, lon | :) = ps(time | :, lat | :, lon | :) * 100
        ps@units = "Pa"
        print(ps@units)
      else
        print(ps@units)
      end if

      ; printVarSummary(T)
      ; printVarSummary(Q)
      ; printVarSummary(ps)

      ; Compute annual mean values of ps
      psavg = ps
      do iy = 0,ntime/12-1
        psavg(time | iy*12:iy*12+11, lat | :, lon | :) = conform(ps(time | iy*12:iy*12+11, lat | :, lon | :), dim_avg_n_Wrap(ps(time | iy*12:iy*12+11, lat | :, lon | :),0), (/1,2/))
      end do

      ssum = cp*T + (Lv*Q)
      copy_VarCoords(T, ssum)

      sum_diff = center_finite_diff_n(ssum, dt, False, 0, 0)
      sum_diff_moist = center_finite_diff_n(Lv*Q, dt, False, 0, 0)

      ptop = min(lev)

      dp = dpres_plevel_Wrap(lev, psavg, ptop, 0)      ; time, lev, lat, lon
      storage = (/ dim_sum_n_Wrap(sum_diff*dp, 1) /)   ; time, lat, lon
      storage = storage / g                            ; time, lat, lon
      copy_VarCoords(ps, storage)
      ; printVarSummary(storage)

      storagemoist = (/ dim_sum_n_Wrap(sum_diff_moist*dp, 1) /)  ; time, lat, lon
      storagemoist = storagemoist / g                            ; time, lat, lon
      copy_VarCoords(ps, storagemoist)
      ; printVarSummary(storagemoist)

      filo = dirup+modelname(m)+"/"+"storage_Amon_"+modelname(m)+"_"+casename(c)+"_"+variantid(m)+"_1x1.25_"+timeperiod(m)+".nc"
      system("/bin/rm -f " + filo)
      fout = addfile(filo, "c")
      fout->lat = pslat
      fout->lon = pslon
      fout->storage = storage
      delete(fout)

      filo = dirup+modelname(m)+"/"+"storagemoist_Amon_"+modelname(m)+"_"+casename(c)+"_"+variantid(m)+"_1x1.25_"+timeperiod(m)+".nc"
      system("/bin/rm -f " + filo)
      fout = addfile(filo, "c")
      fout->lat = pslat
      fout->lon = pslon
      fout->storagemoist = storagemoist
      delete(fout)

      delete(f)
      delete(lat)
      delete(lon)
      delete(tlat)
      delete(tlon)
      delete(pslat)
      delete(pslon)
      delete(lev)
      delete(T)
      delete(Q)
      delete(ps)
      delete(psavg)
      delete(ssum)
      delete(sum_diff)
      delete(ptop)
      delete(dp)
      delete(storage)
      delete(storagemoist)
    end do
end do
end
