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
from glob import glob

import numpy as np
import pandas as pd
from astropy.time import Time
from tqdm import tqdm
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
import xarray as xa
# import sys
# sys.path.append('/home/muscat/muscat2/')
# from toi_functions import get_toi

#http://www.oao.nao.ac.jp/en/telescope/abouttel188/
oao = EarthLocation.from_geodetic(lat='34:34:37.47', lon='133:35:38.24', height=372*u.m)

class MuscatReader:
    def __init__(self, obsdate, objname, objcoord, bands=['g','r','z_s'], ref_frame=0,
                 photdir='apphot_mapping', datadir= '/ut2/muscat/reduction/muscat',
                 verbose=True, overwrite=False):
        """initialize
        """
        self.obsdate = obsdate
        self.objname = objname
        self.bands = bands
        self.ref_frame = ref_frame
        self.photdir = photdir
        self.datadir = datadir
        self.obj_coord = self._get_obj_coord(objcoord)
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
        obj_coord = SkyCoord(ra=objcoord[0], dec=objcoord[1], unit='deg')
        return obj_coord

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
            errmsg1 = f'{path} is empty'
            assert len(radius_dirs)>0, errmsg1
            paths[band] = radius_dirs
            nradii[band] = (len(radius_dirs))
        errmsg2 = f'nradii: {nradii} have unequal lengths'
        assert len(set(nradii.values()))==1, errmsg2
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
                previous_shape = []
                for i,dat_file in enumerate(dat_files):
                    #parse first line which contains time
                    try:
                        d = pd.read_csv(dat_file, header=None)
                        time = float(d.iloc[0].str.split('=').values[0][1]) #gjd - 2450000
                        time+=2450000
                        # parse 18th and 20th line
                        exptime.append(float(d.iloc[18].str.split('=').values[0][1]))
                        airmass.append(float(d.iloc[20].str.split('=').values[0][1]))
                    except Exception as e:
                        #some afphot dat files may be corrupted
                        errmsg = f'{dat_file} seems corrupted.\n'
                        errmsg+='You can temporarily delete the radius directory in each band:\n'
                        for n,_ in enumerate(self.bands):
                            p = f'{self.datadir}/{self.obsdate}/{self.objname}_{n}/{self.photdir}/rad{radius}\n'
                            errmsg+=f'$ rm -rf {p}'
                        raise IOError(errmsg)
                    # parse succeeding lines as dataframe
                    d = pd.read_csv(dat_file, delim_whitespace=True, comment='#', names=column_names)

                    previous_shape.append(d.shape)
                    if i>1:
                        errmsg = f'{dat_file} has shape {d.shape} instead of {previous_shape[i-1]}\n'
                        errmsg+='You can temporarily delete the radius directory in each band:\n'
                        for n,_ in enumerate(self.bands):
                            p = f'{self.datadir}/{self.obsdate}/{self.objname}_{n}/{self.photdir}/rad{radius}\n'
                            errmsg+=f'$ rm -rf {p}'
                        assert previous_shape[i-1]==d.shape, errmsg
                    assert len(d)>0, f'{dat_file} seems empty'
                    cadences[time]=d
                #save each data frame corresponding to each cadence/ exposure sequence
                assert len(cadences)>0, f'{cadences} seems empty'
                apertures[radius] = cadences
            assert len(apertures)>0, f'{apertures} seems empty'
            data[band] = apertures
            airmasses[band] = airmass #not band-dependent but differs in length
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
                                                 ra=self.obj_coord.ra.deg,
                                                 dec=self.obj_coord.dec.deg,
                                                 lat=oao.lat.deg,
                                                 longi=oao.lon.deg,
                                                 alt=oao.height.value)
                bjds[band] = result[0]

            else:
                #BJD time format in TDB time scale
                bjds[band] = (jd.tdb + jd.light_travel_time(self.obj_coord)).value

                #check difference between two time scales (should be < 8 mins!)
                diff=bjds[band]-2400000.5-mjds[band]
                diff_in_minutes = np.median(diff)*24*60
                assert diff_in_minutes < 8.4, f'{band}: {diff_in_minutes:.2} min'
        self.mjds = mjds
        self.bjds = bjds
        #return mjds, bjds

    def create_cpix_xarray(self, band):
        """pixel centroids of each star in a given reference frame
        """
        jd = self.jds[band][self.ref_frame]
        radius = self.radii[band][0] #any radius will do
        d = self.data[band][radius][jd]
        cpix = xa.DataArray(d[['xcen','ycen']],
                            name='centroids_pix',
                            dims='star centroid_pix'.split(),
                            coords={'centroid_pix': ['x', 'y'],
                                    'star': d['ID']})
        return cpix

    def create_csky_xarray(self, band):
        """just place-holder for sky centroids since not available in afphot
        (or as if astrometry.net failed in M2 pipeline)
        """
        cpix = self.create_cpix_xarray(band)
        ca = np.full_like(np.array(cpix), np.nan)
        csky = xa.DataArray(ca,
                            name='centroids_sky',
                            dims='star centroid_sky'.split(),
                            coords={'centroid_sky': ['ra', 'dec'],
                                    'star': cpix.star.data})
        return csky

    def create_flux_xarray(self, band):
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

        # container
        fluxes = np.zeros((ncadences,napers,nstars))
        # populate
        for n,jd in enumerate(self.jds[band]):
            for m,r in enumerate(self.radii[band]):
                fluxes[n,m] = d[r][jd]['flux'].values
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

    def create_eobj_xarray(self, band):
        """fwhm as proxy to object entropy (eobj)
        """
        d = self.data[band]
        r = self.radii[band][0] #any radius will do
        jd = self.jds[band][0]
        stars = d[r][jd]['ID'].values
        nstars = len(stars)
        apers = self.radii[band]
        napers = len(apers)
        ncadences = len(self.jds[band])

        # container
        eobjs = np.zeros((ncadences,napers,nstars))
        # populate
        for n,jd in enumerate(self.jds[band]):
            for m,r in enumerate(self.radii[band]):
                eobjs[n,m] = d[r][jd]['fwhm'].values
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

    def create_msky_xarray(self, band):
        """sky as proxy to sky median (msky)
        """
        d = self.data[band]
        r = self.radii[band][0]
        jd = self.jds[band][0]
        stars = d[r][jd]['ID'].values
        nstars = len(stars)
        napers = len(self.radii[band])
        ncadences = len(self.jds[band])

        # container
        mskys = np.zeros((ncadences,nstars))
        # populate
        for n,jd in enumerate(self.jds[band]):
            for m in range(nstars):
                mskys[n,m] = d[r][jd].iloc[m]['sky']

        # construct
        msky = xa.DataArray(mskys,
                            name='msky',
                            dims='mjd star'.split(),
                            coords={'mjd': self.mjds[band],
                                    'star': stars
                                   })
        return msky

    def create_esky_xarray(self, band):
        """peak as proxy to sky entropy (esky)
        """
        d = self.data[band]
        r = self.radii[band][0]
        jd = self.jds[band][0]
        stars = d[r][jd]['ID'].values
        nstars = len(stars)
        napers = len(self.radii[band])
        ncadences = len(self.jds[band])

        # container
        eskys = np.zeros((ncadences,nstars))
        # populate
        for n,jd in enumerate(self.jds[band]):
            for m in range(nstars):
                eskys[n,m] = d[r][jd].iloc[m]['peak']

        # construct
        esky = xa.DataArray(eskys,
                            name='esky',
                            dims='mjd star'.split(),
                            coords={'mjd': self.mjds[band],
                                    'star': stars
                                   })
        return esky

    def create_centroid_xarray(self, band):
        """pixel centroids with shape (time,nstars,['x','y'])
        """
        d = self.data[band]
        r = self.radii[band][0]
        jd = self.jds[band][0]
        stars = d[r][jd]['ID'].values
        nstars = len(stars)
        ncadences = len(self.jds[band])
        axis = ['x', 'y']
        naxis = len(axis)

        # container
        centroids = np.zeros((ncadences,nstars,naxis))
        # populate
        for n,jd in enumerate(self.jds[band]):
            for m in range(nstars):
                centroids[n,m] = d[r][jd].iloc[m][['xcen','ycen']]

        # construct
        centroid = xa.DataArray(centroids,
                            name='centroid',
                            dims='mjd star axis'.split(),
                            coords={'mjd': self.mjds[band],
                                    'axis': axis,
                                    'star': stars
                                   })
        return centroid

    def create_aux_xarray(self, band):
        """auxiliary variables: mjd, exptime, airmass
        which are all parsed from commented rows (0, 18, 20) in each dat file
        """
        quantities = ['airmass', 'exptime', 'mjd']
        naux = len(quantities)
        ncadences = len(self.mjds[band])
        auxs = np.c_[self.airmasses[band], self.exptimes[band], self.mjds[band]]

        # construct
        aux = xa.DataArray(auxs,
                            name='aux',
                            dims='frame quantity'.split(),
                            coords={'quantity': quantities,
                                    #'frame':
                                   })
        return aux

    def create_dataset(self, band):
        """
        """
        cpix = self.create_cpix_xarray(band)
        csky = self.create_csky_xarray(band)
        flux = self.create_flux_xarray(band)
        eobj = self.create_eobj_xarray(band)
        msky = self.create_msky_xarray(band)
        esky = self.create_esky_xarray(band)
        cpos = self.create_centroid_xarray(band)
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

    def save_as_nc(self, overwrite=None, outdir='.'):
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
