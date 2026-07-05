# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
# One-time setup
git submodule update --init --recursive
cd external/cochlea-1 && export CFLAGS="-std=c99" && python setup.py build_ext --inplace && cd ../..

mamba create -n sl_env -c conda-forge python=3.9.23 nest-simulator=3.8.0 -f environment.yml
mamba activate sl_env   # always activate before running anything
```

## Running the Pipeline

The pipeline has three sequential stages, each a separate entry point.

### 1. NEST Spiking Simulation
```bash
cd simulate && python main.py
```
No CLI args — edit parameters directly in `main.py` lines ~60–86 (sound, model, cochlea, angles). Outputs dill-pickled `.pic` files to `RESULTS/`.

### 2. LFP Reconstruction
```bash
# Single process
python LFP_reconstruction/main_reconstruct.py \
  --pic-file RESULTS/<file>.pic --angle 0 --side L --n-cells 100

# MPI (production)
mpiexec -n 4 python LFP_reconstruction/main_reconstruct.py \
  --pic-file RESULTS/<file>.pic --angle 0 --side L --n-cells 15500 --n-single 5
```
Key flags: `--monaural` (silence contralateral SBC), `--hoc-file` (passive vs active MSO morphology), `--y-track` (probe rostrocaudal track index 0–4).

### 3. ABR Reconstruction
```bash
mpiexec -n 4 python ABR_reconstruction/main_abr.py \
  --pic-file RESULTS/<file>.pic --angle 0 --side L --n-cells 200 \
  --electrodes Cz A1 A2 --condition binaural
```

## Architecture & Data Flow

```
Sound stimulus
    → HRTF binaural filtering (hrtf_utils.run_hrtf, IRCAM SOFA dataset)
    → Cochlear model → 35,000 ANF spike trains (3,500 CF × 10 fibers)
          [ZilanyCochlea / GammatoneCochlea / TanCarneyCochlea]
          [cached in data/ANF_SPIKETRAINS/<model>/]
    → NEST simulation (BrainstemModel.py)
          ANF → SBC/GBC → MNTBC/LNTBC → MSO / LSO / SPN
          iaf_cond_beta neurons; custom x_to_one tonotopic connector
          saved to RESULTS/*.pic (dill)
    → LFP reconstruction (hybridLFPy + NEURON)
          MSOPopulation: tonotopic spike assignment, Exp2Syn synapses
          NEURON morphology: MSO_models/mso_model[_active].hoc
          80-channel probe (5 rostrocaudal × 16 mediolateral)
          saved to RESULTS/lfp_tmp/output_*/
    → ABR reconstruction (4-sphere head model)
          MSO dipole → scalp potentials at Cz, A1, A2 (µV)
          saved to RESULTS/abr_tmp/output_*/
```

## Key Files

| File | Role |
|------|------|
| `simulate/main.py` | Top-level entry: configure & run NEST sim |
| `simulate/models/BrainstemModel/BrainstemModel.py` | Network topology, populations (MSO 15,500/side), x_to_one connector |
| `simulate/models/BrainstemModel/params.py` | All neuronal/synaptic parameters as a dataclass |
| `simulate/utils/hrtf_utils.py` | HRTF loading, ITD computation, binaural processing |
| `simulate/utils/custom_sounds.py` | Sound classes: Tone, ToneBurst, Click, WhiteNoise, HarmonicComplex |
| `LFP_reconstruction/main_reconstruct.py` | MSOPopulation subclass + full LFP pipeline |
| `LFP_reconstruction/extract_spikes.py` | .pic → GDF spike files consumed by hybridLFPy |
| `ABR_reconstruction/main_abr.py` | CurrentDipoleMoment → 4-sphere scalp model |
| `MSO_models/mso_model.hoc` | Passive MSO morphology (soma + two dendrites) |
| `MSO_models/mso_model_active.hoc` | Active MSO (adds Klt and Ih channels) |

## Coordinate Convention

- **x** — dorsoventral / tonotopic axis (`ELLIPSE_RADIUS_X = 443 µm`)
- **y** — rostrocaudal axis / probe channel direction (`ELLIPSE_RADIUS_Y = 2845 µm`)
- **z** — mediolateral / dendritic / probe-depth axis (probe spans −400 to +400 µm; all MSO somas at z = 0)

Probe is at `x = −30 µm`, `y` per track, channels along z. All neurons rendered at `z = 0` (flat cylinder).

## .pic File Structure

```python
{
  'sounds': {'base_sound', 'gated_sound', 'l_hrtf_sounds', 'r_hrtf_sounds'},
  'angle_to_rate': {
      angle: {
          'L': {'ANF': {times, senders, global_ids}, 'SBC': {...}, 'MSO': {...}, ...},
          'R': {...}
      }
  },
  'conf': {'parameters', 'cochlea_type', 'sound_key', 'model_desc'},
  'simulation_time': 100,   # ms
  'times': {start, end, timetaken}
}
```

## Populations & Sizes (per side)

| Population | N | Role |
|---|---|---|
| ANF | 35,000 | Auditory nerve fibers (cochlea output) |
| SBC | 28,000 | Spherical bushy cells → MSO excitation (ipsi) |
| GBC | 3,600 | Globular bushy cells → MNTBC/LNTBC drive |
| MNTBC | 3,600 | Medial NTB → MSO inhibition (contra) |
| LNTBC | 3,600 | Lateral NTB → LSO inhibition |
| MSO | 15,500 | Medial superior olive (binaural coincidence) |
| LSO | 5,600 | Lateral superior olive (ILD coding) |
| SPN | 3,600 | Superior paraolivary nucleus |

## Output Locations

- `RESULTS/*.pic` — main simulation results
- `RESULTS/lfp_tmp/spikes_*/` — GDF spike files
- `RESULTS/lfp_tmp/output_*/figures/` — LFP plots (colourmap, single cells, neurophonic)
- `RESULTS/abr_tmp/output_*/` — ABR.h5, population_dipole.h5, figures
- `data/ANF_SPIKETRAINS/{Zilany,Gammatone,TanCarney}/` — joblib-cached ANF responses
- `logs/` — execution logs per script run
