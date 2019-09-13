# Cluster SZ

This is a CosmoSIS module for modeling the [Sunyzev-Zel'dovich](https://ui.adsabs.harvard.edu/abs/1972CoASP...4..173S/abstract)
signal around galaxy clusters and groups, using the [cluster\_toolkit](https://github.com/jhod0/cluster_toolkit)
package.

The Sunyaev-Zel'dovich effect is a modulation of the Cosmic Microwave Background
caused by inverse Compton-scattering of CMB photons off of hot gas within a
galaxy cluster (or elsewhere). This module predicts the Compton-y parameter, the
observable analog of the SZ effect. It is a direct measure of the projected
pressure profile of gas within galaxy clusters, and therefore closely related
to the structure and composition of clusters.

This module computes both one- and two-halo contributions to the Compton-y
profile, following
[Hill et al.](https://ui.adsabs.harvard.edu/abs/2018PhRvD..97h3501H/abstract)
and
[Vikram et al.](https://ui.adsabs.harvard.edu/abs/2017MNRAS.467.2315V/abstract).
The one-halo term currently implements the
[Battaglia et al.](https://ui.adsabs.harvard.edu/abs/2012ApJ...758...75B/abstract)
profile, but we plan to add implementations of the modified profiles presented
in Hill.


## Installation

TODO

## Usage

TODO
