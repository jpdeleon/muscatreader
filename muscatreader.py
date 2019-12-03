#!/usr/bin/env python
"""
The script converts the .dat files from afphot to .nc files for M2 pipeline.

Before running this script, afphot should be ran (usually in muscat-abc)
and its results copied to /ut2/muscat/reduction/muscat/DATE.

To convert .dat to .nc, this script does the following.
1. read the .dat files in /ut2/muscat/reduction/muscat/DATE/TARGET_N/PHOTDIR/radXX.0
where
    DATE: observation date (e.g. 191029)
    TARGET_N: e.g. TOI516_0, TOI516_1, TOI516_2 for g-,r-,z-band produced by afphot
    PHOTDIR: either apphot_mapping or apphot_centroid
    radXX.0: radius containing .dat files
2. convert JD to BJD_TDB, although M2 pipeline uses MJD_TDB
3. construct xarrays assuming:
    fwhm as proxy to object entropy (eobj)
    sky as proxy to sky median (msky)
    peak as proxy to sky entropy (esky)
3. save xarrays dataset into .nc files for each band
"""
import os
import re
from glob import glob
import pandas as pd
from astropy.time import Time
from tqdm import tqdm
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
import xarray as xa
import matplotlib.pyplot as pl
from astroplan.plots import plot_finder_image
from astropy.visualization import ZScaleInterval
interval = ZScaleInterval(contrast=0.5)

from muscat2ph.phdata import PhotometryData
# import sys
# sys.path.append('/home/muscat/muscat2/')
# from toi_functions import get_toi

#http://www.oao.nao.ac.jp/en/telescope/abouttel188/
oao = EarthLocation.from_geodetic(lat='34:34:37.47', lon='133:35:38.24', height=372*u.m)

muscat_fov = 6.8 #arcsec in diagonal
fov_rad = muscat_fov*u.arcmin 
interval = ZScaleInterval()

def binned(a, binsize, fun=np.mean):
    a_b = []
    for i in range(0, a.shape[0], binsize):
        a_b.append(fun(a[i:i+binsize], axis=0))
        
    return a_b

class DatReader:
    def __init__(self, obsdate, objname, objcoord, bands=['g','r','z_s'], nstars=None, 
                 ref_frame=0, ref_band='r',
                 photdir='apphot_mapping', datadir= '/ut2/muscat/reduction/muscat', 
                 verbose=True, overwrite=False):
        """initialize
        """
        if 'z' in bands:
            raise ValueError('use z_s instead of z')
        self.obsdate = obsdate
        self.objname = objname
        self.bands = bands
        self.ref_band = ref_band
        self.ref_frame = ref_frame
        self.nstars = nstars
        self.photdir = photdir
        self.datadir = datadir
        self.objcoord = self._get_obj_coord(objcoord)
        self.paths = self._get_paths()
        self.airmasses = None
        self.exptimes = None
        self.data = self._load_dat_files()
        self.radii = {band: sorted(self.data[band].keys()) for band in self.bands}
        self.jds = {band: sorted(self.data[band][self.radii[band][0]].keys()) for band in self.bands}
        self.mjds = None 
        self.bjds = None #tdb
        self.use_barycorrpy = False
        self._convert_to_bjd_tdb() #populate mjds and bjds attributes
        self.verbose = verbose
        self.overwrite = overwrite
        
    def _get_obj_coord(self, objcoord):
        """Define coord used in bjd_tdb conversion
        """
        objcoord = SkyCoord(ra=objcoord[0], dec=objcoord[1], unit='deg')
        return objcoord
    
    def _get_paths(self):
        """get path to each data directory per band
        """
        paths = {}
        nradii = {}
        
        loc = f'{self.datadir}/{self.obsdate}'
        if not os.path.exists(loc):
            raise FileNotFoundError(f'afphot files not found in {loc}')
            
        for n,band in enumerate(self.bands):
            path = f'{loc}/{self.objname}_{n}/{self.photdir}'
            radius_dirs = glob(path+'/rad*')
            errmsg = f'{path} is empty'
            assert len(radius_dirs)>0, errmsg
            paths[band] = radius_dirs
            nradii[band] = (len(radius_dirs))
        errmsg = f'nradii: {nradii} have unequal number of radius directories'
        assert len(set(nradii.values()))==1, errmsg
        return paths
    
    def _load_dat_files(self):
        """get data per band per aperture radius per cadence;
        aperture radius is parsed from the directory produced by afphot
        
        Note: aperture radius in afphot is chosen arbitrarily,
        whereas M2 pipeline uses 9 radii: (4,8,12,16,20,25,30,40,50) pix
        
        TODO: when a .dat file is corrupted, it is better to populate
        the entry with a dataframe of null/NaN values; currrently it is
        simpler to omit/skip using the entire radius directory
        """
        data = {}
        exptimes = {}
        airmasses = {}
        for band in tqdm(self.bands, desc='reading .dat files'):
            radius_dirs = self.paths[band]

            apertures = {}
            for radius_dir in radius_dirs:
                #parse radius from directory name
                radius = float(radius_dir.split('/')[-1][3:])
                #get dat files inside aperture radius directory
                dat_files = glob(radius_dir+'/*')
                dat_files.sort()
                #specify column names based written in .dat file
                column_names = 'ID xcen ycen nflux flux err sky sky_sdev SNR nbadpix fwhm peak'.split()

                cadences = {}
                exptime = []
                airmass = []
                nrows, ncols = [], []
                for i,dat_file in enumerate(dat_files):
                    try:
                        #parse lines 0, 18, 20 which contains gjd, exptime, and airmass
                        d = pd.read_csv(dat_file, header=None)
                        time = float(d.iloc[0].str.split('=').values[0][1]) #gjd - 2450000
                        time+=2450000
                        exptime.append(float(d.iloc[18].str.split('=').values[0][1]))
                        airmass.append(float(d.iloc[20].str.split('=').values[0][1]))
                    except Exception as e:
                        #some afphot dat files may be corrupted
                        errmsg = f'{dat_file} seems corrupted.\n\n'
                        errmsg+='You can temporarily delete the radius directory in each band:\n'
                        for n,_ in enumerate(self.bands):
                            p = f'{self.datadir}/{self.obsdate}/{self.objname}_{n}/{self.photdir}/rad{radius}\n'
                            errmsg+=f'$ rm -rf {p}'
                        raise IOError(errmsg)
                        
                    # parse succeeding lines as dataframe  
                    d = pd.read_csv(dat_file, delim_whitespace=True, comment='#', names=column_names)
                    nrows.append(d.shape[0])
                    ncols.append(d.shape[1])
                    
                    # in cases when data is bad, the number of stars detected 
                    # in some frames is less than nstars used in afphot;
                    if (self.nstars is not None) and self.nstars < len(d):
                        # trim to fewer stars
                        d = d.iloc[:self.nstars]
                    
                    # check if each .dat file has same shape as the rest
                    nrow = int(np.median(nrows))
                    ncol = int(np.median(ncols))
                    if i>1:              
                        errmsg =f'{dat_file} has shape {d.shape} instead of {(nrow,ncol)}\n\n'
                        errmsg+=f'You can set nstars<={d.shape[0]}, or\n\n'
                        errmsg+='You can temporarily delete the radius directory in each band:\n'
                        for n,_ in enumerate(self.bands):
                            p = f'{self.datadir}/{self.obsdate}/{self.objname}_{n}/{self.photdir}/rad{radius}\n'
                            errmsg+=f'$ rm -rf {p}'
                        assert (nrow,ncol)==d.shape, errmsg
                    assert len(d)>0, f'{dat_file} seems empty'
                    cadences[time]=d
                #save each data frame corresponding to each cadence/ exposure sequence
                assert len(cadences)>0, f'{cadences} seems empty'
                apertures[radius] = cadences
            assert len(apertures)>0, f'{apertures} seems empty'
            data[band] = apertures
            airmasses[band] = airmass #not band-dependent but differs in length per band
            exptimes[band] = exptime
        #set attributes
        self.exptimes = exptimes
        self.airmasses = airmasses
        return data
    
    def _convert_to_bjd_tdb(self):
        """convert jd to bjd format and tdb time scale
        """
        mjds = {}
        bjds = {}
        for band in self.bands:
            radius = self.radii[band][0] #any radius will do
            d = self.data[band][radius] #because time per radius is identical 
            jd = Time(list(d.keys()), format='jd', scale='utc', location=oao)
            #mjd time format
            mjds[band] = jd.mjd
                
            if self.use_barycorrpy:
                #https://arxiv.org/pdf/1801.01634.pdf
                try:
                    from barycorrpy import utc_tdb
                except:
                    raise ImportError("pip install barycorrpy")
                #convert jd to bjd_tdb
                result = utc_tdb.JDUTC_to_BJDTDB(jd, 
                                                 ra=self.objcoord.ra.deg, 
                                                 dec=self.objcoord.dec.deg, 
                                                 lat=oao.lat.deg, 
                                                 longi=oao.lon.deg, 
                                                 alt=oao.height.value)
                bjds[band] = result[0]
                
            else:
                #BJD time format in TDB time scale
                bjds[band] = (jd.tdb + jd.light_travel_time(self.objcoord)).value
                
                #check difference between two time scales (should be < 8 mins!)
                diff=bjds[band]-2400000.5-mjds[band]
                diff_in_minutes = np.median(diff)*24*60
                assert diff_in_minutes < 8.4, f'{band}: {diff_in_minutes:.2} min'
        self.mjds = mjds
        self.bjds = bjds
        #return mjds, bjds
        
    def show_ref_table(self):
        """return dat file table corresponding to ref_frame
        """
        radius = self.radii[self.ref_band][0] #any radius will do
        jds = self.jds[self.ref_band]
        df = self.data[self.ref_band][radius][jds[self.ref_frame]]
        return df
    
    def create_cpix_xarray(self, dummy_value=None):
        """pixel centroids of each star in a given reference frame and band
        
        Note that cpix is the same for all bands
        """
        ref_jd = self.jds[self.ref_band][self.ref_frame]
        radius = self.radii[self.ref_band][0] #any radius will do
        d = self.data[self.ref_band][radius][ref_jd]
        if dummy_value is None:
            cen = d[['xcen','ycen']]
        else:
            cen = np.full(d[['xcen','ycen']].shape, dummy_value)
            
        cpix = xa.DataArray(cen,
                            name='centroids_pix',
                            dims='star centroid_pix'.split(),
                            coords={'centroid_pix': ['x', 'y'],
                                    'star': d['ID']})
        return cpix
    
    def create_csky_xarray(self):
        """just place-holder for sky centroids since not available in afphot
        (or as if astrometry.net failed in M2 pipeline)
        """
        cpix = self.create_cpix_xarray()
        ca = np.full_like(np.array(cpix), np.nan) 
        csky = xa.DataArray(ca,
                            name='centroids_sky',
                            dims='star centroid_sky'.split(),
                            coords={'centroid_sky': ['ra', 'dec'],
                                    'star': cpix.star.data})
        return csky
        
    def create_flux_xarray(self, band, dummy_value=None):
        """flux with shape (time,napertures,nstars)
        """
        d = self.data[band]
        r = self.radii[band][0]
        jd = self.jds[band][0]
        stars = d[r][jd]['ID'].values
        nstars = len(stars)
        apers = self.radii[band]
        napers = len(apers)
        ncadences = len(self.jds[band])
        
        # populate
        if dummy_value is None:
            fluxes = np.zeros((ncadences,napers,nstars))
            for n,jd in enumerate(self.jds[band]):
                for m,r in enumerate(self.radii[band]):
                    fluxes[n,m] = d[r][jd]['flux'].values
        else:
            fluxes = np.full((ncadences,napers,nstars), dummy_value)
        #reshape
        fluxes = fluxes.reshape(-1,nstars,napers)
        # fluxes.shape

        # construct
        flux = xa.DataArray(fluxes,
                            name='flux',
                            dims='mjd star aperture'.split(),
                            coords={'mjd': self.mjds[band],
                                    'aperture': apers,
                                    'star': stars
                                   })
        return flux
    
    def create_eobj_xarray(self, band, dummy_value=None):
        """fwhm as proxy to object entropy (eobj)
        
        Note that entropy e in M2 pipeline is defined as:
        z = (a - a.min() + 1e-10)/ a.sum()
        e = -(z*log(z)).sum()
        """
        d = self.data[band]
        r = self.radii[band][0] #any radius will do
        jd = self.jds[band][0]
        stars = d[r][jd]['ID'].values
        nstars = len(stars)
        apers = self.radii[band]
        napers = len(apers)
        ncadences = len(self.jds[band])
        
        # populate
        if dummy_value is None:
            eobjs = np.zeros((ncadences,napers,nstars))
            for n,jd in enumerate(self.jds[band]):
                for m,r in enumerate(self.radii[band]):
                    eobjs[n,m] = d[r][jd]['fwhm'].values
        else:
            eobjs = np.full((ncadences,napers,nstars), dummy_value)
            
        #reshape
        eobjs = eobjs.reshape(-1,nstars,napers)

        # construct
        eobj = xa.DataArray(eobjs,
                            name='eobj',
                            dims='mjd star aperture'.split(),
                            coords={'mjd': self.mjds[band],
                                    'aperture': apers,
                                    'star': stars
                                   })
        return eobj
    
    def create_msky_xarray(self, band, dummy_value=None):
        """sky as proxy to sky median (msky)
        """
        d = self.data[band]
        r = self.radii[band][0]
        jd = self.jds[band][0]
        stars = d[r][jd]['ID'].values
        nstars = len(stars)
        cadences = self.jds[band]
        ncadences = len(cadences)
        
        # populate
        if dummy_value is None:
            mskys = np.zeros((ncadences,nstars))
            for n,jd in enumerate(cadences):
                for m,star in enumerate(stars):
                    mskys[n,m] = d[r][jd].iloc[m]['sky']
        else:
            mskys = np.full((ncadences,nstars), dummy_value)
        # construct
        msky = xa.DataArray(mskys,
                            name='msky',
                            dims='mjd star'.split(),
                            coords={'mjd': self.mjds[band],
                                    'star': stars
                                   })
        return msky
    
    def create_esky_xarray(self, band, dummy_value=None):
        """sky_sdev as proxy to sky entropy (esky)
        """
        d = self.data[band]
        r = self.radii[band][0]
        jd = self.jds[band][0]
        stars = d[r][jd]['ID'].values
        nstars = len(stars)
        cadences = self.jds[band]
        ncadences = len(cadences)
        
        # populate
        if dummy_value is None:
            eskys = np.zeros((ncadences,nstars))
            for n,jd in enumerate(cadences):
                for m,star in enumerate(stars):
                    eskys[n,m] = d[r][jd].iloc[m]['sky_sdev']
        else:
            eskys = np.full((ncadences,nstars), dummy_value)
            
        # construct
        esky = xa.DataArray(eskys,
                            name='esky',
                            dims='mjd star'.split(),
                            coords={'mjd': self.mjds[band],
                                    'star': stars
                                   })
        return esky
    
    def create_centroid_xarray(self, band, dummy_value=None):
        """pixel centroids with shape (time,nstars,['x','y'])
        """
        d = self.data[band]
        r = self.radii[band][0]
        jd = self.jds[band][0]
        stars = d[r][jd]['ID'].values
        nstars = len(stars)
        cadences = self.jds[band]
        ncadences = len(cadences)
        axis = ['x', 'y']
        naxis = len(axis)
        
        # populate
        if dummy_value is None:
            centroids = np.zeros((ncadences,nstars,naxis))
            for n,jd in enumerate(cadences):
                for m,star in enumerate(stars):
                    centroids[n,m] = d[r][jd].iloc[m][['xcen','ycen']]
        else:
            centroids = np.full((ncadences,nstars,naxis), dummy_value)
        # construct
        centroid = xa.DataArray(centroids,
                            name='centroid',
                            dims='mjd star axis'.split(),
                            coords={'mjd': self.mjds[band],
                                    'axis': axis,
                                    'star': stars
                                   })
        return centroid
    
    def create_aux_xarray(self, band, dummy_value=None):
        """auxiliary variables: jd, exptime, airmass
        which are all parsed from commented rows (0, 18, 20) in each dat file
        
        Note that mjd is used instead of jd 
        """
        quantities = ['airmass', 'exptime', 'mjd']
        naux = len(quantities)
        ncadences = len(self.mjds[band])
        
        if dummy_value is None:
            auxs = np.c_[self.airmasses[band], self.exptimes[band], self.mjds[band]]
        else:
            auxs = np.full((ncadences, naux), dummy_value)
            
        # construct
        aux = xa.DataArray(auxs,
                            name='aux',
                            dims='frame quantity'.split(),
                            coords={'quantity': quantities,
                                    'frame': range(ncadences)
                                   })
        return aux
    
    def create_dataset(self, band):
        """
        join all xarray to create a dataset
        """
        #star pix positions in reference frame
        cpix = self.create_cpix_xarray() 
        #star sky positions in reference frame
        csky = self.create_csky_xarray() 
        flux = self.create_flux_xarray(band, dummy_value=None)
        #entropy object (fwhm)
        eobj = self.create_eobj_xarray(band, dummy_value=None)
        #median sky
        msky = self.create_msky_xarray(band, dummy_value=None)
        #entropy sky (peak)
        esky = self.create_esky_xarray(band, dummy_value=None)
        #centroid positions
        cpos = self.create_centroid_xarray(band, dummy_value=None)
        #auxiliary observables
        aux = self.create_aux_xarray(band)

        ds = xa.Dataset(dict(flux=flux, 
                         obj_entropy=eobj, 
                         sky_median=msky, 
                         sky_entropy=esky, 
                         centroid=cpos,
                         aux=aux, 
                         centroids_pix=cpix, 
                         centroids_sky=csky
                            )
                        )
        return ds
    
    def show_master_flat(self, dataloc='/ut2/muscat/reduction/muscat', figsize=(12,4)):
        """
        show master flat image per band
        """
        nbands = len(self.bands)
        fig, axs = pl.subplots(1, nbands, figsize=figsize)
        ax = axs.flatten()

        for n,band in enumerate(self.bands):
            fp = f'{dataloc}/{self.obsdate}/FLAT/flat/flat_ccd{n}.fits'
            img = fits.getdata(fp)
            hdr = fits.getheader(fp)

            zmin, zmax = interval.get_limits(img)
            ax[n].imshow(img, vmin=zmin, vmax=zmax, origin='bottom')
            ax[n].set_title(f'{band}-band')
            
        return fig
    
    def show_master_dark(self, dataloc='/ut2/muscat/reduction/muscat', figsize=(12,4)):
        """
        show master dark image per band
        """
        nbands = len(self.bands)
        fig, axs = pl.subplots(1, nbands, figsize=figsize)
        ax = axs.flatten()

        for n,band in enumerate(self.bands):
            fp = f'{dataloc}/{self.obsdate}/FLAT/dark/dark_ccd{n}_flat.fits'
            img = fits.getdata(fp)
            hdr = fits.getheader(fp)

            zmin, zmax = interval.get_limits(img)
            ax[n].imshow(img, vmin=zmin, vmax=zmax, origin='bottom')
            ax[n].set_title(f'{band}-band')
            
        return fig
    
    def show_ref_frame(self, tid=0, refid=None, ref_band=None, survey=None,
                       dataloc='/ut2/muscat/reduction/muscat', figsize=(10,8)):
        """
        show star ids superposed with the reference frame
        
        Note: refid refers to fits filename frame number 
        whereas ref_frame starts with 0 for first science frame
        """
        bandnum = {band:i for i,band in enumerate(self.bands)}
        refband = ref_band if ref_band is not None else self.ref_band
        bn = bandnum[refband]
            
        if refid is None:
            path = f'{dataloc}/{self.obsdate}/{self.objname}/region/*.reg'
            region_file_path = glob(path)[0]
            refid = region_file_path.split('.')[-1][:-4]
            reffilename = region_file_path.split('/')[-1].split('.')[0]
        else:
            refid = str(refid).zfill(4)
            reffilename = f'MSCT{bn}_{self.obsdate}{refid}'
            region_file_path = f'{dataloc}/{self.obsdate}/{self.objname}/region/{reffilename}.reg'
        pos = np.loadtxt(region_file_path, dtype='str')

        centers = []
        for n in range(len(pos)-1):
            line= n+1
            i = re.findall(r'\d+', ''.join(pos[line]))
            x=float(f'{i[0]}.{i[1]}')
            y=float(f'{i[2]}.{i[3]}')
            centers.append((x,y))
            
        #
        fp = f'{dataloc}/{self.obsdate}/{self.objname}/reference/ref-{reffilename}.fits'
        img = fits.getdata(fp)
        hdr = fits.getheader(fp)

        fig, ax = pl.subplots(1, 2, figsize=figsize)

        #imaging survey
        if survey is None:
            survey='DSS2 Blue'
            
        #left panel
        zmin, zmax = interval.get_limits(img)
        ax[0].imshow(img, vmin=zmin, vmax=zmax, origin='bottom', cmap='viridis');
        ax[0].set_title(hdr['OBJECT'])
        #circles
        for n,center in enumerate(centers[:self.nstars]):
            if n==tid:
                c = pl.Circle(center, 20, color='r', fill=False, lw=3)
            else:
                c = pl.Circle(center, 20, color='w', fill=False, lw=3)
                
            x,y = center
            ax[0].text(x+5, y+5, n, color='w', fontsize=15)
            ax[0].add_artist(c)

        #right panel
        plot_finder_image(self.objcoord, fov_radius=fov_rad, survey=survey, reticle=True, ax=ax[1]);
        ax[1].set_title(survey)
        return fig
    
    def save_as_nc(self, overwrite=None, outdir='./photometry'):
        """save as .nc file per band
        
        Note that filename does not differentiate whether afphot
        was produced by either centroid or mapping algorithm!
        """
        overwrite = self.overwrite if overwrite is None else overwrite
        
        datasets = {}
        for band in self.bands:
            ds = self.create_dataset(band)
            #save
            outpath = f'{outdir}/{self.obsdate}'
            if not os.path.exists(outpath):
                os.makedirs(outpath)
                
            fp = f'{outpath}/{self.objname}_{self.obsdate}_{band}.nc'
            if os.path.exists(fp) and not overwrite:
                raise FileExistsError(f'{fp} exists! Else, use overwrite=True')
            else:
                ds.to_netcdf(fp)

            if self.verbose:
                print(f'Saved: {fp}')
            
            datasets[band] = ds
        return datasets
            
        
class NcReader():
    def __init__(self, obsdate, objname, tid=0, cids=[1], bands=['g','r','z_s'], 
                 datadir='./photometry', verbose=True, overwrite=False):
        """initialize
        """
        if 'z' in bands:
            raise ValueError('use z_s instead of z')
        self.obsdate = obsdate
        self.objname = objname
        self.bands = bands
        self.tid = tid
        self.cids = cids
        self.datadir = datadir
        self.data = self._load_nc_files()
        self.radii = {band: self.data[band]._ds.aperture.data for band in self.bands}
        self.verbose = verbose
        #self.objcoord = objcoord
    
    def _load_nc_files(self, datadir=None):
        """load saved nc files
        """
        data = {}
        datadir = datadir if datadir is not None else self.datadir 
        for band in self.bands:
            #load nc files
            fp = f'{datadir}/{self.obsdate}/{self.objname}_{self.obsdate}_{band}.nc'
            data[band] = PhotometryData(fp, self.tid, self.cids, objname=self.objname, objskycoords=None)
        return data
        
    def plot_quicklook(self, tid=None, cids=None, bintime=10, aper_radii=None, 
                fluxlim=None, dxlim=None, dylim=None, fwhmlim=None, zlim=None,
                ingress=None, midpoint=None, egress=None,
                save_csv=False, save_png=False, figsize=(15,10)):
        """show quick look similar to lcm in afphot
        """
        tid = tid if tid is not None else self.tid
        cids = cids if cids is not None else self.cids
        assert isinstance(cids, list)
        assert isinstance(aper_radii, list)
        nbands = len(self.bands)
        assert len(aper_radii)==nbands
        
        fig, axs = pl.subplots(5, nbands, figsize=figsize, 
                               gridspec_kw={'height_ratios': [2,1,1,1,1]},
                               #subplot_kw={'hspace': None},
                               sharex=True,
                               constrained_layout=False
                              )
        ax = axs.flatten()
        
        colnames = 'time flux err airmass entropy_sky entropy_obj dx dy'.split()
                
        colors = {'g':'b',
                  'r':'g',
                  'i':'y',
                  'z_s':'r'}
        for n,band in enumerate(self.bands):
            #load nc files
            photdata = self.data[band]
            
            if aper_radii is None:
                #get smallest radius
                radii = self.radii[band]
                radius = radii[0]
            else:
                radii = np.copy(aper_radii)
                radius = radii[n]
                errmsg = f'aper radius not in [{self.radii[band]}]'
                assert radius in self.radii[band], errmsg
            
            #get index of radius
            ridx, = np.where(radius in radii)[0]
                       
            # use methods in PhotometryData
                        
            t = photdata._ds['mjd'].data
            f = photdata.relative_fluxes[:, ridx].data
            e = photdata.relative_error.data
            exp = photdata.aux.data[:,1].mean()
            
            #binning
            cadence = np.median(np.diff(t))*u.day.to(u.second)
            binsize=int(bintime*u.min.to(cadence*u.second))
            t2 = binned(t, binsize=binsize)
            f2 = binned(f, binsize=binsize)
            e2 = binned(e, binsize=binsize)/np.sqrt(binsize)

            airmass = photdata._ds['aux'][:,0].data
            entropy_sky = photdata._ds['sky_entropy'][:, tid].data
            entropy_obj = photdata._ds['obj_entropy'][:, tid, ridx].data
            
            x, y = photdata._ds['centroids_pix'][tid].data
            dx = photdata._ds['centroid'][:, tid, 0].data - x
            dy = photdata._ds['centroid'][:, tid, 1].data - y
            
            with np.errstate(all='ignore'):
                #to suppress errors when comparing <> with nan
                if fluxlim:
                    assert isinstance(fluxlim, tuple)
                    fluxmask = (f>fluxlim[0]) & (f<fluxlim[1])
                    ymin,ymax = fluxlim[0], fluxlim[1]
                else:
                    ymin, ymax = 0.9, 1.1
                    fluxmask = (f>ymin) & (f<ymax)
                if dxlim:
                    assert isinstance(dxlim, tuple)
                    dxmask = (abs(dx) > dxlim[0]) & (abs(dx) < dxlim[1])
                else:
                    dxmask = (abs(dx) > 0) & (abs(dx) < 15)
                if dylim:
                    assert isinstance(dylim, tuple)
                    dymask = (abs(dy) > dylim[0]) & (abs(dy) < dylim[1])
                else:
                    dymask = (abs(dy) > 0) & (abs(dy) < 15)
                if fwhmlim:
                    assert isinstance(fwhmlim, tuple)
                    fwhmmask = (entropy_obj>fwhmlim[0]) & (entropy_obj<fwhmlim[1])
                else:
                    fwhmmask = (entropy_obj>0) & (entropy_obj<10)
                if zlim:
                    assert isinstance(zlim, tuple)
                    zmask = (airmass>zlim[0]) & (airmass<zlim[1])
                else:
                    zmask = (airmass>1.0) & (airmass<2.5)
                
            #apply mask
            idx = fluxmask & dxmask & dymask & fwhmmask & zmask
            t=t[idx]
            f=f[idx]
            e=e[idx]
            airmass=airmass[idx]
            entropy_sky=entropy_sky[idx]
            entropy_obj=entropy_obj[idx]
            dx=dx[idx]
            dy=dy[idx]

            #raw
            ax[n].plot(t,f,'.k',alpha=0.1)
            #binned
            color = colors[band]
            ax[n].errorbar(t2,f2,yerr=e2,fmt='o',markersize=5,color=color,label=f'{bintime}-min bin')
            
            if save_csv:
                fout=f'{self.objname}_{self.obsdate}_r{radius}_{band}_best.csv'
                df = pd.DataFrame(np.c_[t,f,e,airmass,entropy_sky,entropy_obj,dx,dy], columns=colnames)
                df.to_csv(fout, index=False)
                print(f'Saved: {fout}')

            if np.all([ingress,midpoint,egress]):
                ax[n].axvline(ingress.mjd, 0, 1, color='k', linestyle='--')
                ax[n].axvline(midpoint.mjd, 0, 1, color='k', linestyle='--')
                ax[n].axvline(egress.mjd, 0, 1, color='k', linestyle='--')

            ax[n].set_title(f'{band} (r={radius} pix)')
            ax[n].set_ylabel('Flux')
            ax[n].set_ylim(ymin,ymax)
            ax[n].legend()
            ax[n+nbands].plot(t,dx,label='dx')
            ax[n+nbands].plot(t,dy,label='dy')
            ax[n+nbands].legend(title='shift')
            ax[n+nbands*2].plot(t,entropy_obj,label='fwhm proxy')
            ax[n+nbands*2].legend()
            ax[n+nbands*3].plot(t,entropy_sky,label='sky proxy')
            ax[n+nbands*3].legend()
            ax[n+nbands*4].plot(t,airmass,'k-',label='airmass')
            ax[n+nbands*4].legend()
            ax[n+nbands*4].set_xlabel('MJD')
        pl.suptitle(f'{self.objname} ({self.obsdate})', y=1.0, fontsize='large')
        fig.tight_layout()
        
        if save_png:
            fout = f'{self.objname}_{self.obsdate}.png'
            fig.savefig(fout, bbox_inches='tight')
            
        return fig
