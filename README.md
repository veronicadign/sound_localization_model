# Sound Localization

This repository contains a biologically grounded computational framework for simulating the human auditory periphery and brainstem for sound localization. It includes peripheral cochlear models (Zilany 2009/2014, Holmberg 2007, Gammatone, Tan–Carney), HRTF preprocessing, spike-train generation, spiking neural networks (LSO/MSO pathways), and plotting utilities.



## Environment Setup
To create and activate the environment:

mamba create -n name_env -c conda-forge python=3.9.23 nest-simulator=3.8.0 -f env.yml
mamba activate name_env


## External Cochlea Submodule (cochlea-1)
This project uses a custom fork of the auditory periphery package cochlea-1, included as a Git submodule inside:
external/cochlea-1

### Initialize the submodule

```bash
git submodule update --init --recursive
cd external/cochlea-1
export CFLAGS="-std=c99"
python setup.py build_ext --inplace
```

After building, cochlea can be imported from anywhere inside this repository:
from cochlea.zilany2014 import run_zilany2014

## External Data
**IRCAM Dataset [HRTFs]**  
Pre-installed HRTFs from 3 different human subjects are included.  
[Download here](http://recherche.ircam.fr/equipes/salles/listen/download.html)  


## Simulation
To run a simulation:  
1. Modify the input parameters in `main.py`:
   - Sound stimulus  
   - Binaural cue scenario  
   - Cochlea model  
   - Neuronal parameters  
2. Execute the script.  

Logs will be generated to monitor the simulation and help identify any errors.

---

## Results
- After generating ANF spike trains, a directory `ANF_SPIKETRAINS` will be created inside `Data` to store them. These can be reused in future simulations since spike train generation is computationally intensive.  
- A `.pic` file will be saved in the `RESULTS` directory, which is automatically created to store all output data.  

---

## Plots
Jupyter Notebooks to replicate the figures from the paper are included in `src/plot` repository.
