# Crystal Field analysis of SrTb2O4 inelastic neutron scattering data

This repository contains the reduced inelastic neutron scattering data files for SrTb2O4 and Python data analysis scripts for fitting the crystal field parameters as described in the paper _Orlandi et al._, "Magnetic properties of the zigzag ladder compound SrTb2O4".

The Python files require [Mantid](https://www.mantidproject.org/) to run to perform the actual crystal field fitting operation outlined in the paper.

The `mcphase` folder contains the mean-field random-phase-approximation calculation using the [McPhase](https://mcphase.github.io/webpage/) program. In addition, to perform the powder averaging we use [SpinW](https://spinw.org), but this is not needed to run the MF-RPA calculation.


## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work by Manh Duc Le is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
