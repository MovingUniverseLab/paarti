#   Originally written by Tom Murphy
#   Original source code at https://tmurphy.physics.ucsd.edu/astr597/exercises/speckle.html


from math import *
import numpy
import numpy.fft
import sys
from scipy.special import gamma
from numpy.random import normal

def factorial(n):
    return gamma(n+1)

def zernike(j,npix=256,phase=0.0):

    if (j > 820):
        print("For n < 40, pick j < 820")
        sys.exit()

    x = numpy.arange(-npix/2,npix/2,dtype='d')
    y = numpy.arange(-npix/2,npix/2,dtype='d')

    xarr = numpy.outer(numpy.ones(npix,dtype='d'),x)
    yarr = numpy.outer(y,numpy.ones(npix,dtype='d'))

    rarr = numpy.sqrt(numpy.power(xarr,2) + numpy.power(yarr,2))/(npix/2)
    thetarr = numpy.arctan2(yarr,xarr) + phase

    outside = numpy.where(rarr > 1.0)

    narr = numpy.arange(40)
    jmax = (narr+1)*(narr+2)/2
    wh = numpy.where(j <= jmax)
    n = int(wh[0][0])
    mprime = int(j - n*(n+1)/2)
    if ((n % 2) == 0):
        m = int(2*int(floor(mprime/2)))
    else:
        m = int(1 + 2*int(floor((mprime-1)/2)))

    radial = numpy.zeros((npix,npix),dtype='d')
    zarr = numpy.zeros((npix,npix),dtype='d')

    for s in range(int((n-m)/2 + 1)):
        tmp = pow(-1,s) * factorial(n-s)
        tmp /= factorial(s)*factorial((n+m)/2 - s)*factorial((n-m)/2 - s)
        radial += tmp*numpy.power(rarr,n-2*s)

    if (m == 0):
        zarr = radial
    else:
        if ((j % 2) == 0):
            zarr = sqrt(2.0)*radial*numpy.cos(m*thetarr)
        else:
            zarr = sqrt(2.0)*radial*numpy.sin(m*thetarr)

    zarr *= sqrt(n+1)
    zarr[outside] = 0.0

    return zarr

def aperture(npix=256, cent_obs=0.0, spider=0):

    illum = numpy.ones((npix,npix),dtype='d')
    x = numpy.arange(-npix/2,npix/2,dtype='d')
    y = numpy.arange(-npix/2,npix/2,dtype='d')

    xarr = numpy.outer(numpy.ones(npix,dtype='d'),x)
    yarr = numpy.outer(y,numpy.ones(npix,dtype='d'))

    rarr = numpy.sqrt(numpy.power(xarr,2) + numpy.power(yarr,2))/(npix/2)
    outside = numpy.where(rarr > 1.0)
    inside = numpy.where(rarr < cent_obs)

    illum[outside] = 0.0
    if numpy.any(inside[0]):
        illum[inside] = 0.0

    if (spider > 0):
        start = int(npix/2 - int(spider)/2)
        illum[start:start+int(spider),:] = 0.0
        illum[:,start:start+int(spider)] = 0.0

    return illum

def plane_wave(npix=256):

    wf = numpy.zeros((npix,npix),dtype='d')

    return wf

def seeing(d_over_r0, npix=256, nterms=15, level=None, quiet=False):

    scale = pow(d_over_r0,5.0/3.0)

    if (nterms < 10):
        print("C'mon, at least use ten terms...")
        sys.exit()

    if level:
        narr = numpy.arange(400,dtype='d') + 2
        coef = numpy.sqrt(0.2944*scale*(numpy.power((narr-1),-0.866) - numpy.power(narr,-0.866)))
        wh = numpy.where(coef < level)
        n = wh[0][0]
        norder = int(ceil(sqrt(2*n)-0.5))
        nterms = int(norder*(norder+1)/2)
        if (nterms < 15):
            nterms = 15

    wf = numpy.zeros((npix,npix),dtype='d')

    resid = numpy.zeros(nterms,dtype='d')
    coeff = numpy.zeros(nterms,dtype='d')

    resid[0:10] = [1.030,0.582,0.134,0.111,0.088,0.065,0.059,0.053,0.046,0.040]
    if (nterms > 10):
        for i in range(10,nterms):
            resid[i] = 0.2944*pow(i+1,-0.866)

    for j in range(2,nterms+1):
        coeff[j-1] = sqrt((resid[j-2]-resid[j-1])*scale)
        wf += coeff[j-1]*normal()*zernike(j,npix=npix)

    if not quiet:
        print("Computed Zernikes to term %d and RMS %f" % (nterms,coeff[nterms-1]))

    return wf

def psf(aperture, wavefront, overfill=1):

    npix = len(wavefront)
    nbig = int(npix*overfill)
    wfbig = numpy.zeros((nbig,nbig),dtype='d')

    half = int((nbig - npix)/2)
    wfbig[half:half+npix,half:half+npix] = wavefront

    illum = numpy.zeros((nbig,nbig),dtype='d')
    illum[half:half+npix,half:half+npix] = aperture

    phase = numpy.exp(wfbig*(0.+1.j))
    input = illum*phase

    ft = numpy.fft.fft2(input)
    powft = numpy.real(numpy.conj(ft)*ft)

    sorted = numpy.zeros((nbig,nbig),dtype='d')
    slice_nbig = int(nbig/2)
    sorted[:slice_nbig,:slice_nbig] = powft[slice_nbig:,slice_nbig:]
    sorted[:slice_nbig,slice_nbig:] = powft[slice_nbig:,:slice_nbig]
    sorted[slice_nbig:,:slice_nbig] = powft[:slice_nbig,slice_nbig:]
    sorted[slice_nbig:,slice_nbig:] = powft[:slice_nbig,:slice_nbig]

    crop =  sorted[half:half+npix,half:half+npix]

    fluxrat = numpy.sum(crop)/numpy.sum(sorted)
    #print "Cropped PSF has %.2f%% of the flux" % (100*fluxrat)

    return crop

def flux_in(img,ctrx,ctry,rad):

    flux = 0.0
    xp = numpy.outer(numpy.ones(10,dtype='d'),numpy.linspace(-0.45,0.45,10))
    yp = numpy.outer(numpy.linspace(-0.45,0.45,10),numpy.ones(10,dtype='d'))
    for x in range(numpy.size(img,0)):
        for y in range(numpy.size(img,1)):
            r = sqrt(pow(x-ctrx,2) + pow(y-ctry,2))
            if (r - rad < 1.0):
                xgrid = x + xp - ctrx
                ygrid = y + yp - ctry
                rgrid = numpy.sqrt(xgrid*xgrid + ygrid*ygrid)
                whin = numpy.where(rgrid < rad)
                count = len(whin[0])
                flux += img[x][y]*count/100.0

    return flux
