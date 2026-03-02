from inspect import getsource

import nest
import numpy as np
from typing import Union 

from utils.cochlea_utils import AnfResponse
from utils.anf_utils import spikes_to_nestgen

from utils.manual_fixes_to_nest.connect import connect
from utils.log_utils import logger, tqdm

from ..SpikingModel import SpikingModel
from .params import Parameters


class BrainstemModel(SpikingModel):
    name = "Brainstem Model"
    key = "v0"

    def __init__(self, params: Parameters, anf: AnfResponse):
        self.params = params
        logger.debug("Creating spike generator according to input IHC response...")
        anfs_per_ear = spikes_to_nestgen(anf)
        logger.debug(anfs_per_ear)
        self.anf = anf
        logger.debug("Creating rest of network...")
        self.create_network(params, anfs_per_ear)
        logger.debug("Model creation complete. Starting simulation...")

    def describe_model(self):
        return {"name": self.name, "networkdef": getsource(self.create_network)}

    def create_network(self, P: Parameters, anfs_per_ear):
        self.pops = {"L": {}, "R": {}}
        self.recs = {"L": {}, "R": {}}


        for side in ["L", "R"]:

            # ------------------------------------------------------
            # ANFs + parrots
            # ------------------------------------------------------
            self.pops[side]["ANF"] = anfs_per_ear[side]
            self.pops[side]["parrot_ANF"] = nest.Create(
                "parrot_neuron", len(self.pops[side]["ANF"])
            )

            # ------------------------------------------------------
            # SBC
            # ------------------------------------------------------
            self.pops[side]["SBC"] = nest.Create(
                "iaf_cond_beta",
                P.POP_NUM.n_SBCs,
            )

            # ------------------------------------------------------
            # GBC
            # ------------------------------------------------------
            self.pops[side]["GBC"] = nest.Create(
                "iaf_cond_beta",
                P.POP_NUM.n_GBCs,
            )

            # ------------------------------------------------------
            # LNTB inhibitory cells
            # ------------------------------------------------------
            self.pops[side]["LNTBC"] = nest.Create(
                "iaf_cond_beta",
                P.POP_NUM.n_LNTBCs
            )

            # ------------------------------------------------------
            # MNTB principal neurons (calyx)
            # ------------------------------------------------------
            self.pops[side]["MNTBC"] = nest.Create(
                "iaf_cond_beta",
                P.POP_NUM.n_MNTBCs
            )

            # ------------------------------------------------------
            # MSO coincidence detector
            # ------------------------------------------------------
            self.pops[side]["MSO"] = nest.Create(
                "iaf_cond_beta",
                P.POP_NUM.n_MSOs,
            )

            # ------------------------------------------------------
            # LSO (EI comparator)
            # ------------------------------------------------------
            self.pops[side]["LSO"] = nest.Create(
                "iaf_cond_beta",
                P.POP_NUM.n_LSOs
            )

            # ------------------------------------------------------
            # SPN (offset detector)
            # ------------------------------------------------------
            self.pops[side]["SPN"] = nest.Create(
                "iaf_cond_beta",
                P.POP_NUM.n_SPNs,
            )
                
        for side in ["L", "R"]:
            for pop in self.pops[side].keys():
                self.recs[side][pop] = nest.Create("spike_recorder")
                connect(self.pops[side][pop], self.recs[side][pop], "all_to_all")

        # real ANFs (generators) to parrots
        connect(self.pops["R"]["ANF"], self.pops["R"]["parrot_ANF"], "one_to_one", syn_spec={"delay": 0.01})
        connect(self.pops["L"]["ANF"], self.pops["L"]["parrot_ANF"], "one_to_one", syn_spec={"delay": 0.01})


        # ANFs to SBCs
        connect(
            self.pops["R"]["parrot_ANF"],
            self.pops["R"]["SBC"],
            "x_to_one",
            syn_spec={"weight": P.SYN_WEIGHTS.ANFs2SBCs,
                       "delay": P.SYN_DELAYS.ANFs2SBCs},
            num_sources=P.POP_CONV.ANFs2SBCs,
        )
        connect(
            self.pops["L"]["parrot_ANF"],
            self.pops["L"]["SBC"],
            "x_to_one",
            syn_spec={"weight": P.SYN_WEIGHTS.ANFs2SBCs,
                       "delay": P.SYN_DELAYS.ANFs2SBCs},
            num_sources=P.POP_CONV.ANFs2SBCs,
        )

        # ANFs to GBCs
        connect(
            self.pops["R"]["parrot_ANF"],
            self.pops["R"]["GBC"],
            "x_to_one",
            syn_spec={"weight": P.SYN_WEIGHTS.ANFs2GBCs,
                       "delay": P.SYN_DELAYS.ANFs2GBCs},
            num_sources=P.POP_CONV.ANFs2GBCs,
        )

        connect(
            self.pops["L"]["parrot_ANF"],
            self.pops["L"]["GBC"],
            "x_to_one",
            syn_spec={"weight": P.SYN_WEIGHTS.ANFs2GBCs,
                       "delay": P.SYN_DELAYS.ANFs2GBCs},
            num_sources=P.POP_CONV.ANFs2GBCs,
        )

        connect(
            self.pops["R"]["GBC"],
            self.pops["R"]["LNTBC"],
            "x_to_one",
            syn_spec={
                "weight": P.SYN_WEIGHTS.GBCs2LNTBCs,
                "delay": P.SYN_DELAYS.GBCs2LNTBCs,
                },
            num_sources=P.POP_CONV.GBCs2LNTBCs,
        )

        connect(
            self.pops["L"]["GBC"],
            self.pops["L"]["LNTBC"],
            "x_to_one",
            syn_spec={
                "weight": P.SYN_WEIGHTS.GBCs2LNTBCs,
                "delay": P.SYN_DELAYS.GBCs2LNTBCs,
            },
            num_sources=P.POP_CONV.GBCs2LNTBCs,
        )

        connect(
            self.pops["R"]["GBC"],
            self.pops["L"]["MNTBC"],
            "x_to_one",
            syn_spec={
                "weight": P.SYN_WEIGHTS.GBCs2MNTBCs,
                "delay": P.SYN_DELAYS.GBCs2MNTBCs,
            },
            num_sources=P.POP_CONV.GBCs2MNTBCs,
        )

        connect(
            self.pops["L"]["GBC"],
            self.pops["R"]["MNTBC"],
            "x_to_one",
            syn_spec={
                "weight": P.SYN_WEIGHTS.GBCs2MNTBCs,
                "delay": P.SYN_DELAYS.GBCs2MNTBCs,
            },
            num_sources=P.POP_CONV.GBCs2MNTBCs,
        )

        # MSO
        # From SBCs (excitation):
        # r_MSO
        #       ipsi
        connect(
            self.pops["R"]["SBC"],
            self.pops["R"]["MSO"],
            "x_to_one",
            syn_spec={
                "weight": P.SYN_WEIGHTS.SBCs2MSO,
                "delay": P.SYN_DELAYS.SBCs2MSOipsi,
            },
            num_sources=P.POP_CONV.SBCs2MSOs,
        )
        #       contra
        connect(
            self.pops["L"]["SBC"],
            self.pops["R"]["MSO"],
            "x_to_one",
            syn_spec={
                "weight": P.SYN_WEIGHTS.SBCs2MSO,
                "delay": P.SYN_DELAYS.SBCs2MSOcontra,
            },
            num_sources=P.POP_CONV.SBCs2MSOs,
        )
        # l_MSO
        #       ipsi
        connect(
            self.pops["L"]["SBC"],
            self.pops["L"]["MSO"],
            "x_to_one",
            syn_spec={
                "weight": P.SYN_WEIGHTS.SBCs2MSO,
                "delay": P.SYN_DELAYS.SBCs2MSOipsi,
            },
            num_sources=P.POP_CONV.SBCs2MSOs,
        )
        #       contra
        connect(
            self.pops["R"]["SBC"],
            self.pops["L"]["MSO"],
            "x_to_one",
            syn_spec={
                "weight": P.SYN_WEIGHTS.SBCs2MSO,
                "delay": P.SYN_DELAYS.SBCs2MSOcontra,
            },
            num_sources=P.POP_CONV.SBCs2MSOs,
        )
        # From LNTBCs (inhibition), ipsi
        connect(
            self.pops["R"]["LNTBC"],
            self.pops["R"]["MSO"],
            "x_to_one",
            syn_spec={
                "weight": P.SYN_WEIGHTS.LNTBCs2MSO,
                "delay": P.SYN_DELAYS.LNTBCs2MSOipsi,
            },
            num_sources=P.POP_CONV.LNTBCs2MSOs,
        )
        # From MNTBCs (inhibition) contra
        connect(
            self.pops["R"]["MNTBC"],
            self.pops["R"]["MSO"],
            "x_to_one",
            syn_spec={
                "weight": P.SYN_WEIGHTS.MNTBCs2MSO,
                "delay": P.SYN_DELAYS.MNTBCs2MSOcontra,
            },
            num_sources=P.POP_CONV.MNTBCs2MSOs,
        )
        # From LNTBCs (inhibition) ipsi
        connect(
            self.pops["L"]["LNTBC"],
            self.pops["L"]["MSO"],
            "x_to_one",
            syn_spec={
                "weight": P.SYN_WEIGHTS.LNTBCs2MSO,
                "delay": P.SYN_DELAYS.LNTBCs2MSOipsi,
            },
            num_sources=P.POP_CONV.LNTBCs2MSOs,
        )
        # From MNTBCs (inhibition) contra
        connect(
            self.pops["L"]["MNTBC"],
            self.pops["L"]["MSO"],
            "x_to_one",
            syn_spec={
                "weight": P.SYN_WEIGHTS.MNTBCs2MSO,
                "delay": P.SYN_DELAYS.MNTBCs2MSOcontra,
            },
            num_sources=P.POP_CONV.MNTBCs2MSOs,
        )

        # LSO

        connect(
            self.pops["R"]["SBC"],
            self.pops["R"]["LSO"],
            "x_to_one",
            syn_spec={"weight": P.SYN_WEIGHTS.SBCs2LSO,
                       "delay": P.SYN_DELAYS.SBCs2LSO},
            num_sources=P.POP_CONV.SBCs2LSOs,
        )
        
        connect(
            self.pops["L"]["SBC"],
            self.pops["L"]["LSO"],
            "x_to_one",
            syn_spec={"weight": P.SYN_WEIGHTS.SBCs2LSO,
                       "delay": P.SYN_DELAYS.SBCs2LSO},
            num_sources=P.POP_CONV.SBCs2LSOs,
        )
        
        connect(
            self.pops["R"]["MNTBC"],
            self.pops["R"]["LSO"],
            "x_to_one",
            syn_spec={"weight": P.SYN_WEIGHTS.MNTBCs2LSO,
                       "delay": P.SYN_DELAYS.MNTBCs2LSO},
            num_sources=P.POP_CONV.MNTBCs2LSOs, 
        )

        connect(
            self.pops["L"]["MNTBC"],
            self.pops["L"]["LSO"],
            "x_to_one",
            syn_spec={"weight": P.SYN_WEIGHTS.MNTBCs2LSO,
                       "delay": P.SYN_DELAYS.MNTBCs2LSO},
            num_sources=P.POP_CONV.MNTBCs2LSOs, 
        )

        connect(
            self.pops["R"]["MNTBC"],
            self.pops["R"]["SPN"],
            "x_to_one",
            syn_spec={"weight": P.SYN_WEIGHTS.MNTBCs2SPN,
                       "delay": P.SYN_DELAYS.MNTBCs2SPN},
            num_sources=P.POP_CONV.MNTBCs2SPNs, 
        )

        connect(
            self.pops["L"]["MNTBC"],
            self.pops["L"]["SPN"],
            "x_to_one",
            syn_spec={"weight": P.SYN_WEIGHTS.MNTBCs2SPN,
                       "delay": P.SYN_DELAYS.MNTBCs2SPN},
            num_sources=P.POP_CONV.MNTBCs2SPNs, 
        )

    def simulate(self, time: Union[float, int]):
        # split in time chunks
        TIME_PER_CHUNK_TQDM = 50
        chunks = time // TIME_PER_CHUNK_TQDM
        logger.debug(
            f"running simulation for {chunks} chunks of {TIME_PER_CHUNK_TQDM}ms each"
        )

        for chunk in tqdm(
            [
                *([TIME_PER_CHUNK_TQDM] * chunks),
                time - chunks * TIME_PER_CHUNK_TQDM,
            ],
            desc="  ⮡ simulation",
        ):
            nest.Simulate(chunk)
        logger.debug(f"total bio time elapsed: {nest.biological_time}")

    def analyze(self):
        result = {"L": {}, "R": {}}
        for side in self.recs.keys():
            for pop_name, pop_data in self.recs[side].items():
                if (
                    pop_name in self.params.CONFIG.STORE_POPS
                    or not self.params.CONFIG.STORE_POPS
                ):
                    result[side][pop_name] = {
                        **pop_data.get("events"),
                        "global_ids": self.pops[side][pop_name].get("global_id"),
                    }

        return result
