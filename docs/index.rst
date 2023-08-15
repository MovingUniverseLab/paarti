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

In order to tell paarti where MAOS configuration files live (the default ones that
come with the MAOS package), you need to set an evironment variable in your
.zshenv, .cshrc, or .bash_profile files. For example, in .zshenv, add:

.. code-block::
   
   export MAOS_CONFIG=/g/lu/code/maos/config



paarti Module
=============

.. toctree::
  :maxdepth: 2

  paarti/index.rst
