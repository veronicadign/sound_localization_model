from dataclasses import dataclass, field
from utils.cochlea_utils import ITD_REMOVAL_STRAT, NUM_CF, NUM_ANF_PER_HC

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
    n_ANFs: int = NUM_CF * NUM_ANF_PER_HC  # Total ANFs in model
    E_L: float = -65.0      # Physiologically realistic resting potential for CN/MSO neurons ~−60…−65 mV
    V_reset: float = -67.0   # Slightly below rest to mimic brief after-hyperpolarization

    # Reversal potentials
    EXC_REV = 0.0
    INH_REV = -75.0

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
    n_MNTBCs: int = n_GBCs
    n_LNTBCs: int = n_GBCs
    n_LSOs: int = n_GBCs
    n_MSOs: int = n_GBCs
    n_LSOs: int = n_GBCs

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


    # --- Synaptic weights (nS or relative units) -----------------
    # ------------------------------------------------------------
    @dataclass
    class SYN_WEIGHTS:
        ANFs2SBCs: float = 15.0      
        ANFs2GBCs: float = 7.0      
        GBCs2LNTBCs: float = 30.0
        GBCs2MNTBCs: float = 30.0
        SBCs2LSO: float = 8.0        
        MNTBCs2LSO: float = -20.0 
        MNTBCs2MSO: float = -20.0 
        LNTBCs2MSO: float = -12.0 
        SBCs2MSO: float = 5.0  

    # ------------------------------------------------------------
    # --- Membrane capacitances (pF) ------------------------------
    # ------------------------------------------------------------
    @dataclass
    class MEMB_CAPS:
        SBC: float = 26.0 
        GBC: float = 26.0   
        MNTBC: float = 20.0
        LNTBC: float = 20.0
        MSO: float = 20.0
        LSO: float = 20.0

    # ------------------------------------------------------------
    # --- Leak conductances (nS) ---------------------------------
    # ------------------------------------------------------------
    @dataclass
    class G_LEAK:
        SBC: float = 34.6   
        GBC: float = 104.0   
        LNTBC: float = 28.0   
        MNTBC: float = 28.0   
        MSO: float = 28.0   
        LSO: float = 28.0   

    # ------------------------------------------------------------
    # --- POPULATION-SPECIFIC THRESHOLDS ---------------------
    # ------------------------------------------------------------
    @dataclass
    class V_TH:
        SBC: float = -45.0     
        GBC: float = -45.0     
        LNTBC: float = -45.0     
        MNTBC: float = -45.0     
        MSO: float = -45.0     
        LSO: float = -45.0     

    # ------------------------------------------------------------
    # --- POPULATION-SPECIFIC REFRACTORY PERIODS (ms) -------
    # ------------------------------------------------------------
    @dataclass
    class T_REF:
        SBC: float = 0.6       
        GBC: float = 0.6       
        LNTBC: float = 0.5     
        MNTBC: float = 0.5     
        MSO: float = 0.5       
        LSO: float = 0.5     

    # ------------------------------------------------------------
    # --- Synaptic time constants (ms) ----------------------------
    # ------------------------------------------------------------
    @dataclass
    class TAUS_EX_RISE:
        SBC: float = 0.2       
        GBC: float = 0.2       
        LNTBC: float = 0.2     
        MNTBC: float = 0.2     
        MSO: float = 0.2       
        LSO: float = 0.2  
    # ------------------------------------------------------------ 
    @dataclass
    class TAUS_EX_DECAY:
        SBC: float = 0.5       
        GBC: float = 0.5       
        LNTBC: float = 0.2     
        MNTBC: float = 0.2     
        MSO: float = 0.2       
        LSO: float = 0.2  
    # ------------------------------------------------------------
    #     @dataclass
    class TAUS_IN_RISE:
        SBC: float = 0.2       
        GBC: float = 0.2       
        LNTBC: float = 0.2     
        MNTBC: float = 0.2     
        MSO: float = 0.2       
        LSO: float = 0.2  
    # ------------------------------------------------------------  
    #     @dataclass
    class TAUS_IN_DECAY:
        SBC: float = 0.2       
        GBC: float = 0.2       
        LNTBC: float = 0.2     
        MNTBC: float = 0.2     
        MSO: float = 0.2       
        LSO: float = 0.2  
    # ------------------------------------------------------------   


    # ------------------------------------------------------------
    # --- System / kernel configuration ---------------------------
    # ------------------------------------------------------------
    @dataclass
    class CONFIG:
        STORE_POPS: set = field(default_factory=lambda: set([]))
        NEST_KERNEL_PARAMS: dict = field(
            default_factory=lambda: {
                "resolution": 0.01,
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
        self.TAUS_EX_RISE = self.TAUS_EX_RISE()
        self.TAUS_EX_DECAY = self.TAUS_EX_DECAY()
        self.TAUS_IN_RISE = self.TAUS_IN_RISE()
        self.TAUS_IN_DECAY = self.TAUS_IN_DECAY()
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
