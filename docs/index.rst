=============
Documentation
=============

This is the documentation for PAARTI, which consists of a the paarti python package
and some helpful MAOS configuration files for different telescopes. MAOS is an
adaptive optics simulation package hosted `here <https://github.com/lianqiw/maos>`__.
Typically, one would run MAOS first to generate some PSFs and then use the 
paarti python package to analyze those PSFs. 

MAOS Configurations
===================
The PAARTI package includes MAOS configuration files for a few different AO systems,
including those at the Keck observatory. We have provided supplementary documentation
for MAOS configuration parameters in a 
`Google Doc <https://docs.google.com/document/d/1_rIU7ttHSZIgGWyOPOsU-dFmtxIsOk-FxezcGy6LKXY/edit?usp=sharing>`__.

paarti Module
=============

.. toctree::
  :maxdepth: 2

  paarti/index.rst

.. note:: The layout of this directory is simply a suggestion.  To follow
          traditional practice, do *not* edit this page, but instead place
          all documentation for the package inside ``paarti/``.
          You can follow this practice or choose your own layout.
