Background:
The dipole instability model numerical code is found in DIM.py. The code is written in python using the NUMBA
just-in-time (JIT) compiler. To install NUMBA, as well as the other required packages, run (preferrably in a venv)

pip install -r requirements.txt

In DIM.py, only the numerical code can be found. The resulting observables, primarily the total dipole moment
given by d+x, can be extracted from the variables output. It should be noted that the simulation time scales 
quadratically with the number of timesteps, so calculations rapidly become tedious for longer simulation times.
The rectangle-rule convolution function can offer a slight speedup in these cases, with only a minor decrease
in accuracy.

It is clear from the preamble that a warning is being silenced. This warning specifies that the s.p. energy
calculation could be carried out more efficiently if a more cumbersome indexing scheme was adopted, and 
presents an unmeasurably small performance penalty.

All variables should have meanings that are clear from chapter 8 of my thesis.

Installation:
The installation process, utilizing a virtual environment, is given by:

git clone https://github.com/DaleHughes766/Dipole_instability_model.git
cd Dipole_instability_model
python3 -m venv .dipole_instability_model          
source .dipole_instability_model/bin/activate
pip install -r requirements.txt
python3 DIM.py
