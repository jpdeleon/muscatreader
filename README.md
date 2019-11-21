# muscatreader
Convert .dat from `afphot` to .nc for MuSCAT2 transit analysis pipeline  

# Introduction
The script converts the .dat files from `afphot` to .nc files for MuSCAT2 pipeline.

Before running this script, `afphot` should be ran (usually in muscat-abc)
and its results copied to /ut2/muscat/reduction/muscat/DATE.

To convert .dat to .nc, this script does the following.
1. read the .dat files in /ut2/muscat/reduction/muscat/DATE/TARGET_N/PHOTDIR/radXX.0
where
    DATE: observation date, e.g. 191029
    TARGET_N: target directories, e.g. TOI516_0, TOI516_1, TOI516_2 for g-,r-,z-bands produced by `afphot`
    PHOTDIR: either `apphot_mapping` or `apphot_centroid` directory
    radXX.0: radius directory containing .dat files
2. convert JD to BJD_TDB, although M2 pipeline uses MJD_TDB
3. construct xarrays assuming:
    fwhm as proxy to object entropy (eobj)
    sky as proxy to sky median (msky)
    peak as proxy to sky entropy (esky)
3. save xarray dataset into .nc files for each band

## Example
```python
#instantiate
mr = MuscatReader(obsdate=191029,
                  objname='TOI516',
                  objcoord=(112.382, 2.848),
                  bands=['g','r','z_s'],
                  ref_frame=20,
                  datadir='/ut2/muscat/reduction/muscat/',
                  photdir='apphot_mapping')
# save .nc in datadir
datasets = mr.save_as_nc(overwrite=True)              
```

Then you can MuSCAT2 `TransitAnalysis` and `TFOPAnalysis` pipeline as usual.
