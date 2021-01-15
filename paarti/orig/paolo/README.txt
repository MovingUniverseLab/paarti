psf.py

A collection of functions to analyze the performance of an OOMAO simulation of
GIRMOS. It generates an oversampled (5x) long-exposure PSF using the function 'wfr_to_psf' with the stack of
wavefront residuals, then it also calculates the diffraction-limited PSF of the
Gemini telescope. It then measures several "empiric" metrics with the function
'psf_metrics'.

Metrics:
- SR (line 729):
  Compare the peak of the PSF with the peak of the flux-normalized
  diffraction-limited PSF.
- FWHM (line 734):
  Find the contour of the PSF at half maximum, then fit it ith an ellipse. It
  returns a major axis FWHM, minor axis FWHM, arithmetic mean FWHM, geometric
  mean FWHM, ellipticity, and position angle (rotation). The fitting of the
  ellipse is done with the 'photutils' package. But, because the fitting can
  fail, I had to use a loop to catch errors and try with different initial
  guesses.
- Encircled/ensquared energy (line 807):
  Calculates the radius of different fractions of EE. It also calculates the
  fraction of EE within different radii.
- Noise equivalent area (line 924):
  It represents the area of detector over which the background introduces a
  noise equivalent to that of a point-like background-dominated source. For this
  case, the NEA is directly proportional to the exposure time required to reach
  a given SNR (see King, 1983).

  Another function that I use often is 'psf_diff', to subtract the central
  (best) PSF from every other PSF in the grid, showing where the light within
  the PSF has been "moved" to and from (see Turri et al., SPIE 2020, Figure 1b).