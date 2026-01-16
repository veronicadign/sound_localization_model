from dataclasses import dataclass, field
from utils.cochlea_utils import ITD_REMOVAL_STRAT

@dataclass
class Parameters:
    key: str = "default_params"

    cochlea: dict[str, dict[str, float]] = field(
        default_factory=lambda: (
            {
                "gammatone": {
                    "hrtf_params": {
                        "subj_number": 0,
                        "ild_only": False,
                        "itd_remove_strategy": ITD_REMOVAL_STRAT.COMPUTED,
                        "apply_gating": True,
                        "ramp_ms": 10,
                    },
                    "noise_factor": 0.3,
                    "refractory_period": 1,  # ms
                    "amplif_factor": 7,
                },
                "TanCarney": {
                    "hrtf_params": {
                        "subj_number": 0,
                        "itd_remove_strategy": ITD_REMOVAL_STRAT.ESTIMATE_FROM_HRTF,
                        "apply_gating": True,
                        "ramp_ms": 10,
                    },
                    "cochlea_params": None,
                    "rng_seed": 42,
                    "omni_noise_level": 0,
                },
                "Zilany": {
                    "hrtf_params": {
                        "subj_number": 0,
                        "itd_remove_strategy": ITD_REMOVAL_STRAT.ESTIMATE_FROM_HRTF,
                    },
                    "cochlea_params": {
                        "anf_num": (6, 2, 2),            # Example fiber counts (HSR, MSR, LSR)
                        "species": "human",
                        "cohc": 1.0,
                        "cihc": 1.0,
                        "powerlaw": "approximate",
                        "ffGn": False
                    },
                    "rng_seed": 42,
                    "omni_noise_level": 0,
                }
            }
        )
    )

    # ------------------------------------------------------------
    # --- GLOBAL membrane & voltage parameters -------------------
    # ------------------------------------------------------------
    n_ANFs: int = 35000
    V_m: float = -65.0       # 🟢 UPDATED (was -70.0)
    # Physiologically realistic resting potential for CN/MSO neurons ~−60…−65 mV
    V_reset: float = -67.0   # 🟢 UPDATED (was same as V_m)
    # Slightly below rest to mimic brief after-hyperpolarization

    # ------------------------------------------------------------
    # --- Population counts --------------------------------------
    # ------------------------------------------------------------
    @dataclass
    class POP_CONN:
        ANFs2SBCs: int = 4
        ANFs2GBCs: int = 20

    SBCs2MSOs: int = int(POP_CONN.ANFs2GBCs / POP_CONN.ANFs2SBCs)
    SBCs2LSOs: int = int(POP_CONN.ANFs2GBCs / POP_CONN.ANFs2SBCs)
    n_SBCs: int = int(n_ANFs / POP_CONN.ANFs2SBCs)
    n_GBCs: int = int(n_ANFs / POP_CONN.ANFs2GBCs)
    n_MSOs: int = n_GBCs
    n_LSOs: int = n_GBCs
    n_inhMSOs: int = n_GBCs

    # ------------------------------------------------------------
    # --- Delays (ms) --------------------------------------------
    # ------------------------------------------------------------
    @dataclass
    class DELAYS:
        GBCs2MNTBCs: float = 0.45
        GBCs2LNTBCs: float = 0.45
        SBCs2MSO_exc_ipsi: float = 2.0
        SBCs2MSO_exc_contra: float = 2.0

        def __init__(self):
            self._DELTA_IPSI: float = 0.2
            self._DELTA_CONTRA: float = -0.4

        @property
        def DELTA_IPSI(self): return self._DELTA_IPSI
        @DELTA_IPSI.setter
        def DELTA_IPSI(self, value): self._DELTA_IPSI = value

        @property
        def DELTA_CONTRA(self): return self._DELTA_CONTRA
        @DELTA_CONTRA.setter
        def DELTA_CONTRA(self, value): self._DELTA_CONTRA = value

        @property
        def LNTBCs2MSO_inh_ipsi(self): return 1.44 + self.DELTA_IPSI
        @property
        def MNTBCs2MSO_inh_contra(self): return 1.44 + self.DELTA_CONTRA

    # ------------------------------------------------------------
    # --- Synaptic time constants (ms) ----------------------------
    # ------------------------------------------------------------
    @dataclass
    class MSO_TAUS:
        rise_ex: float = 0.12   # 🟢 UPDATED (was 0.2)  — fast AMPA
        rise_in: float = 0.3    # 🟢 UPDATED (was 0.2)  — glycinergic
        decay_ex: float = 0.35  # 🟢 UPDATED (was 0.5)
        decay_in: float = 4.0   # 🟢 UPDATED (was 1.5) -> can be shortened more

    # ------------------------------------------------------------
    # --- Synaptic weights (nS or relative units) -----------------
    # ------------------------------------------------------------
    @dataclass
    class SYN_WEIGHTS:
        ANFs2SBCs: float = 35.0      # same
        ANFs2GBCs: float = 7.0       # same
        GBCs2LNTBCs: float = 20.0
        GBCs2MNTBCs: float = 30.0
        SBCs2LSO: float = 8.0        # same
        MNTBCs2LSO: float = -20.0 
        MNTBCs2MSO: float = -12.0 # 🟢 UPDATED (was -40) 
        LNTBCs2MSO: float = -12.0 # 🟢 UPDATED (was -40) 
        SBCs2MSO: float = 12.0  # 🟢 UPDATED (was +9) 

    # ------------------------------------------------------------
    # --- Membrane capacitances (pF) ------------------------------
    # ------------------------------------------------------------
    @dataclass
    class MEMB_CAPS:
        SBC: float = 30.0    # 🟢 UPDATED (was 15) — typical SBC soma 20–40 pF
        GBC: float = 40.0    # 🟢 UPDATED (was 15)
        MNTBC: float = 25.0  # 🟢 UPDATED (was 15)
        LNTBC: float = 25.0  # 🟢 UPDATED (was 15)
        MSO: float = 20.0    # ✅ kept (literature 15–25 pF)
        LSO: float = 35.0    # 🟢 UPDATED (was 30)

    # ------------------------------------------------------------
    # --- Leak conductances (nS) ---------------------------------
    # ------------------------------------------------------------
    @dataclass
    class G_LEAK:
        SBC: float = 20.0   # 🟢 UPDATED (was 40) τm≈1.5 ms with C=30 pF
        GBC: float = 20.0   # 🟢 UPDATED (was 25) τm≈2 ms
        LNTBC: float = 20.0 # 🟢 UPDATED (was 25)
        MNTBC: float = 12.0 # 🟢 UPDATED (was 25) τm≈2 ms, fast relay
        MSO: float = 57.0   # 🟢 UPDATED (was 80) τm≈0.35 ms, fits physiology
        LSO: float = 8.75   # 🟢 UPDATED (was 20) τm≈4 ms (C=35 pF)

    # ------------------------------------------------------------
    # --- NEW: POPULATION-SPECIFIC THRESHOLDS ---------------------
    # ------------------------------------------------------------
    @dataclass
    class V_TH:
        SBC: float = -42.0     # 🟢 NEW — bushy cells fire ~-40/-45
        GBC: float = -40.0     # 🟢 NEW — globular are slightly lower
        LNTBC: float = -45.0   # 🟢 NEW — inhibitory interneurons
        MNTBC: float = -45.0   # 🟢 NEW — fast relay
        MSO: float = -38.0     # 🟢 NEW — MSO threshold ~ -35/-40
        LSO: float = -45.0     # 🟢 NEW — LSO principal cells

    # ------------------------------------------------------------
    # --- NEW: POPULATION-SPECIFIC REFRACTORY PERIODS (ms) -------
    # ------------------------------------------------------------
    @dataclass
    class T_REF:
        SBC: float = 0.5       # 🟢 NEW — bushy cells recover fast
        GBC: float = 0.5       # 🟢 NEW
        LNTBC: float = 0.5     # 🟢 NEW
        MNTBC: float = 0.5     # 🟢 NEW — calyx relay is fast
        MSO: float = 0.2       # 🟢 NEW — sub-ms integration window
        LSO: float = 1.0       # 🟢 NEW — LSO slower membrane


    # ------------------------------------------------------------
    # --- System / kernel configuration ---------------------------
    # ------------------------------------------------------------
    @dataclass
    class CONFIG:
        STORE_POPS: set = field(default_factory=lambda: set([]))
        NEST_KERNEL_PARAMS: dict = field(
            default_factory=lambda: {
                "resolution": 0.1,
                "rng_seed": 42,
                "total_num_virtual_procs": 32,
            }
        )

    # ------------------------------------------------------------
    # --- Post-init to instantiate nested dataclasses -------------
    # ------------------------------------------------------------
    def __post_init__(self):
        self.CONFIG = self.CONFIG()
        self.DELAYS = self.DELAYS()
        self.SYN_WEIGHTS = self.SYN_WEIGHTS()
        self.POP_CONN = self.POP_CONN()
        self.MSO_TAUS = self.MSO_TAUS()
        self.MEMB_CAPS = self.MEMB_CAPS()
        self.G_LEAK = self.G_LEAK()

"""
iaf_cond_alpha default params
{'C_m': 250.0, -> always too big. try with a sensible 10pF
 'Ca': 0.0,
 'E_L': -70.0,
 'E_ex': 0.0,
 'E_in': -85.0,
 'I_e': 0.0,
 'V_m': -70.0,
 'V_reset': -60.0,
 'V_th': -55.0, -> maybe we can try -57... 'might help maintain selectivity for coincident inputs in MSO while still allowing LSO to respond to intensity differences'???
 'archiver_length': 0,
 'available': (0,),
 'beta_Ca': 0.001,
 'capacity': (0,),
 'dg_ex': 0.0,
 'dg_in': 0.0,
 'element_type': 'neuron',
 'elementsize': 688,
 'frozen': False,
 'g_L': 16.6667, -> try 166.67 to compensate for higher C_m
 'g_ex': 0.0,
 'g_in': 0.0,
 'global_id': 0,
 'instantiations': (0,),
 'local': True,
 'model': 'iaf_cond_alpha',
 'model_id': 33,
 'node_uses_wfr': False,
 'post_trace': 0.0,
 'recordables': ('g_ex', 'g_in', 't_ref_remaining', 'V_m'),
 'synaptic_elements': {},
 't_ref': 2.0,
 't_spike': -1.0,
 'tau_Ca': 10000.0,
 'tau_minus': 20.0,
 'tau_minus_triplet': 110.0,
 'tau_syn_ex': 0.2,
 'tau_syn_in': 2.0,
 'thread': -1,
 'thread_local_id': -1,
 'type_id': 'iaf_cond_alpha',
 'vp': -1}
"""
