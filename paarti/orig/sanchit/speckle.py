#!/usr/bin/env python

from math import *
import numpy
import wavefront
from PIL import Image
from astropy.io import fits as pyfits
import os
import matplotlib.pyplot as plt

N = 512						# number of pixels across airy ring
#   Average seeing at Mauna Kea is 0.43 arcsec FWHN
#   r0 is related to seeing as 0.98*lambda/(seeing FWHM[radians])
#   At H Band, lambda = 1.65 microns, we get r0=0.77 m

D_over_r0 = 8.076				# seeing is  lambda/r0 radians

wf = wavefront.seeing(D_over_r0,nterms=15,npix=N,level=0.02)	# get wavefront

wfmax = numpy.max(wf)
wfmin = numpy.min(wf)
wfrng = wfmax - wfmin

print(wfmin,wfmax)				# reported in radians

# create GIF Image of wavefront
im = Image.new('L',(N,N))
for i in range(N):
    for j in range(N):
        x = (i,j)
        im.putpixel(xy=x,value=int(127.0 + wf[i][j]*256./wfrng))
im.save('wavefront.png')

plot_psf = plt.figure()
wav_ax = plot_psf.add_subplot(221)
wav_ax.set_title('Wavefront')
im_plot = wav_ax.imshow(im)

# establish aperture illumination pattern
#illum = wavefront.aperture(npix=N, cent_obs = 0.3,spider=6)
illum = wavefront.aperture(npix=N, cent_obs = 0.0,spider=0)

# create GIF Image of aperture illumination
im = Image.new('L',(N,N))
for i in range(N):
    for j in range(N):
        im.putpixel((i,j),int(illum[i][j]*255))
im.save('aperture.png')
ap_ax = plot_psf.add_subplot(222)
ap_ax.set_title('Aperture')
im_plot = ap_ax.imshow(im)

psf_scale = 5	# pads aperture by this amount; resultant pix scale is

# lambda/D/psf_scale, so for instance full frame 256 pix
# for 3.5 m at 532 nm is 256*5.32e-7/3.5/3 = 2.67 arcsec
# for psf_scale = 3

#   H-band - 1.65*10**-6/10.*256/3 = 2.90 arc seconds

# generate speckle pattern given my wavefront and aperture illumination
final_im = Image.new('L',(N,N))
total_iter = 1

for ii in range(total_iter):
    psf = wavefront.psf(illum,wf,overfill=psf_scale)

    psfmax = numpy.max(psf)
    psfmin = numpy.min(psf)
    psfrng = psfmax - psfmin

# create GIF Image of  speckle pattern

    im = Image.new('L',(N,N))
    for i in range(N):
        for j in range(N):
            im.putpixel((i,j),int(63.0 + psf[i][j]*192/psfrng))
    final_im = numpy.add(im,final_im)

final_av = numpy.divide(final_im,float(total_iter))
psfmax = numpy.max(final_av)
psfmin = numpy.min(final_av)

#im.save('psf.png')
#int_ax = plot_psf.add_subplot(222)
#int_ax.set_title('Integrated Speckles')
#im_plot = int_ax.imshow(final_im[450:550,450:550])

psf_ax = plot_psf.add_subplot(223)
psf_ax.set_title('Speckle Average')
im_plot = psf_ax.imshow(final_av[:,:], vmin=psfmin, vmax=psfmax)


# create FITS Image of speckle pattern
if 'psf.fits' in os.listdir('.'):
    os.unlink('psf.fits')
hdu = pyfits.PrimaryHDU(psf)
hdulist = pyfits.HDUList([hdu])
hdulist.writeto('psf.fits')

# make a plane wavefront and create telescope diffraction-limited pattern
pwf = wavefront.plane_wave(npix=N)
diffrac = wavefront.psf(illum,pwf,overfill=psf_scale)

diffracmax = numpy.max(diffrac)
diffracmin = numpy.min(diffrac)
diffracrng = diffracmax - diffracmin

# create GIF Image of diffraction pattern
im = Image.new('L',(N,N))
for i in range(N):
    for j in range(N):
        im.putpixel((i,j),int(63.0 + diffrac[i][j]*1920/diffracrng))
im.save('diffrac.png')
dif_ax = plot_psf.add_subplot(224)
dif_ax.set_title('Diffraction pattern')
im_f = numpy.asarray(im)
im_plot = dif_ax.imshow(im_f)

# create FITS Image of diffraction pattern
if 'diffrac.fits' in os.listdir('.'):
    os.unlink('diffrac.fits')
hdu = pyfits.PrimaryHDU(diffrac)
hdulist = pyfits.HDUList([hdu])
hdulist.writeto('diffrac.fits')

# print some flux stats
influx = numpy.sum(illum)
outflux = influx*pow(N*psf_scale,2)
print("Cropped PSF has %.2f%% of the flux; cropped airy has %.2f%%" % \
      (numpy.sum(psf)/outflux*100,numpy.sum(diffrac)/outflux*100))
flux = wavefront.flux_in(diffrac,N/2,N/2,1.22*psf_scale)
print("Inside first ring: ",flux/outflux)
#print numpy.sum(illum)
plot_psf.tight_layout()
plot_psf.savefig('Speckles_int_' + str(total_iter) + '.png')
plt.show()