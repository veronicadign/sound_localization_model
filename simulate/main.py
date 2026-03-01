# ==========================================================
# PARAMS GUI FOR NEST BRAINSTEM MODEL
# Run with:
#     python params_GUI.py
# ==========================================================

import nest
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from simulate.models.BrainstemModel.params import Parameters as params

# ==========================================================
# LOAD PARAMETERS
# ==========================================================

P = params()

populations = ["SBC", "GBC", "MNTBC", "LNTBC", "LSO", "MSO", "SPN"]

all_params = {}
for pop in populations:
    all_params[pop] = {
        "C_m":          getattr(P.MEMB_CAPS,      pop),
        "g_L":          getattr(P.G_LEAK,          pop),
        "E_L":          getattr(P.E_L,             pop),
        "V_m":          getattr(P.E_L,             pop),
        "V_reset":      getattr(P.V_RESET,         pop),
        "V_th":         getattr(P.V_TH,            pop),
        "t_ref":        getattr(P.T_REF,           pop),
        "E_ex":         getattr(P.EXC_REV,         pop),
        "tau_rise_ex":  getattr(P.TAUS_EX_RISE,    pop),
        "tau_decay_ex": getattr(P.TAUS_EX_DECAY,   pop),
        "E_in":         getattr(P.INH_REV,         pop),
        "tau_rise_in":  getattr(P.TAUS_IN_RISE,    pop),
        "tau_decay_in": getattr(P.TAUS_IN_DECAY,   pop),
    }

# ==========================================================
# HELPER
# ==========================================================

def make_pre_spike_times(base_times, num_inputs, delta=0.0):
    return [[t + i * delta for t in base_times] for i in range(num_inputs)]

# ==========================================================
# SIMULATION
# ==========================================================

def simulate(population, spike_string, n_ex, n_in, weight_ex, weight_in, delay, neuron_params):
    nest.ResetKernel()
    nest.SetKernelStatus(P.CONFIG.NEST_KERNEL_PARAMS)

    spike_times = [float(x.strip()) for x in spike_string.split(",")]

    post = nest.Create("iaf_cond_beta", 1, params=neuron_params)
    sr   = nest.Create("spike_recorder")
    mm   = nest.Create("multimeter", 1, {
        "record_from": ["V_m", "g_ex", "g_in"],
        "record_to":   "memory",
        "interval":    0.01
    })
    nest.Connect(post, sr)
    nest.Connect(mm,   post)

    if n_ex > 0:
        ex_times = make_pre_spike_times(spike_times, n_ex)
        pres_ex  = nest.Create("spike_generator", len(ex_times),
                               params=[{"spike_times": st} for st in ex_times])
        nest.Connect(pres_ex, post, syn_spec={"weight": weight_ex, "delay": delay})

    if n_in > 0:
        in_times = make_pre_spike_times(spike_times, n_in)
        pres_in  = nest.Create("spike_generator", len(in_times),
                               params=[{"spike_times": st} for st in in_times])
        nest.Connect(pres_in, post, syn_spec={"weight": weight_in, "delay": delay})

    nest.Simulate(10.0)

    data        = nest.GetStatus(mm, "events")[0]
    times       = data["times"]
    V_m         = data["V_m"]
    g_ex        = data["g_ex"]
    g_in        = data["g_in"]
    I_ex        = g_ex * (V_m - neuron_params["E_ex"])
    I_in        = g_in * (V_m - neuron_params["E_in"])
    post_spikes = nest.GetStatus(sr, "events")[0]["times"]

    return times, V_m, g_ex, g_in, I_ex, I_in, post_spikes

# ==========================================================
# GUI
# ==========================================================

class App:
    def __init__(self, root):
        self.root = root
        root.title("NEST Brainstem Model")

        # --------------------------------------------------
        # Right panel: plot  (build first so we know its size)
        # --------------------------------------------------
        self.fig, self.axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        self.fig.tight_layout(pad=1.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # --------------------------------------------------
        # Left panel: scrollable controls, height-locked to plot
        # --------------------------------------------------
        ctrl_outer = ttk.Frame(root)
        ctrl_outer.grid(row=0, column=0, sticky="ns")

        self._canvas = tk.Canvas(ctrl_outer, width=260, highlightthickness=0)
        self._sb     = ttk.Scrollbar(ctrl_outer, orient="vertical",
                                     command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._sb.set)
        self._sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        ctrl = ttk.Frame(self._canvas, padding=(8, 6))
        self._win_id = self._canvas.create_window((0, 0), window=ctrl, anchor="nw")

        ctrl.bind("<Configure>", lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e:
            self._canvas.itemconfig(self._win_id, width=e.width))

        # lock canvas height to plot widget height
        self.canvas.get_tk_widget().bind("<Configure>", lambda e:
            self._canvas.configure(height=e.height))

        # mousewheel
        self._canvas.bind_all("<MouseWheel>", lambda e:
            self._canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # --------------------------------------------------
        # Controls content
        # --------------------------------------------------

        # population
        ttk.Label(ctrl, text="Population").pack(anchor="w")
        self.pop_var = tk.StringVar(value=populations[0])
        pop_menu = ttk.Combobox(ctrl, textvariable=self.pop_var,
                                values=populations, state="readonly")
        pop_menu.pack(fill="x")
        pop_menu.bind("<<ComboboxSelected>>", self.on_pop_change)

        # spike times (update on Enter or FocusOut)
        ttk.Label(ctrl, text="Spike times (ms)").pack(anchor="w", pady=(5, 0))
        self.spike_var = tk.StringVar(value="2")
        spike_entry = ttk.Entry(ctrl, textvariable=self.spike_var)
        spike_entry.pack(fill="x")
        spike_entry.bind("<Return>",    lambda e: self.run())
        spike_entry.bind("<FocusOut>",  lambda e: self.run())

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=5)

        # N excitatory / inhibitory as spinboxes
        self.n_ex = self._spinbox(ctrl, "N excitatory", default=2)
        self.n_in = self._spinbox(ctrl, "N inhibitory", default=0)

        # connection sliders
        self.w_ex  = self._slider(ctrl, "w_ex",          0,   50,  5)
        self.w_in  = self._slider(ctrl, "w_in",        -50,    0, -5)
        self.delay = self._slider(ctrl, "Delay",         0.1,  5,  1)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=5)
        ttk.Label(ctrl, text="Neuron parameters",
                  font=("", 9, "bold")).pack(anchor="w")

        self.neuron_frame = ttk.Frame(ctrl)
        self.neuron_frame.pack(fill="x")
        self.neuron_sliders = {}
        self._build_neuron_sliders(populations[0])

        # --------------------------------------------------
        # Initial plot
        # --------------------------------------------------
        self.run()

    # -------------------------------------------------------
    def _slider(self, parent, label, lo, hi, default, integer=False):
        """Compact slider: label + scale + value on one tight row."""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=1)

        ttk.Label(frame, text=label, width=13, anchor="w").pack(side="left")

        var = tk.DoubleVar(value=default)
        sl  = ttk.Scale(frame, from_=lo, to=hi, variable=var, orient="horizontal")
        sl.pack(side="left", fill="x", expand=True)

        lbl = ttk.Label(frame, width=7,
                        text=str(int(default)) if integer else f"{default:.2f}")
        lbl.pack(side="left")

        def _update(_=None):
            v = round(var.get()) if integer else var.get()
            lbl.config(text=str(v) if integer else f"{v:.2f}")

        def _release(_=None):
            _update()
            self.run()

        sl.bind("<Motion>",          _update)
        sl.bind("<ButtonRelease-1>", _release)
        return var

    def _spinbox(self, parent, label, default=0):
        """Integer entry with +/- buttons, triggers run on change."""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=1)
        ttk.Label(frame, text=label, width=13, anchor="w").pack(side="left")
        var = tk.IntVar(value=default)
        sb  = ttk.Spinbox(frame, from_=0, to=99, textvariable=var, width=5)
        sb.pack(side="left")
        var.trace_add("write", lambda *_: self.root.after_idle(self.run))
        return var

    def _build_neuron_sliders(self, pop):
        for w in self.neuron_frame.winfo_children():
            w.destroy()
        self.neuron_sliders.clear()
        for name, val in all_params[pop].items():
            lo = val - abs(val) if val != 0 else -100
            hi = val + abs(val) if val != 0 else  100
            self.neuron_sliders[name] = self._slider(
                self.neuron_frame, name, lo=lo, hi=hi, default=val)

    def on_pop_change(self, _=None):
        self._build_neuron_sliders(self.pop_var.get())
        self.run()

    # -------------------------------------------------------
    def run(self):
        try:
            neuron_params = {k: v.get() for k, v in self.neuron_sliders.items()}

            times, V_m, g_ex, g_in, I_ex, I_in, post_spikes = simulate(
                population    = self.pop_var.get(),
                spike_string  = self.spike_var.get(),
                n_ex          = self.n_ex.get(),
                n_in          = self.n_in.get(),
                weight_ex     = self.w_ex.get(),
                weight_in     = self.w_in.get(),
                delay         = self.delay.get(),
                neuron_params = neuron_params,
            )

            ylabels = ["g_ex", "g_in", "I_ex / I_in", "V_m"]
            for ax, lbl in zip(self.axes, ylabels):
                ax.clear()
                ax.set_ylabel(lbl, fontsize=8)
                ax.tick_params(labelsize=7)

            self.axes[0].plot(times, g_ex,  color="tab:blue")
            self.axes[1].plot(times, g_in,  color="tab:orange")
            self.axes[2].plot(times, I_ex,  color="tab:blue",   label="I_ex")
            self.axes[2].plot(times, I_in,  color="tab:orange", label="I_in")
            self.axes[2].legend(fontsize=6, loc="upper right")
            self.axes[3].plot(times, V_m,   color="tab:green")
            if len(post_spikes):
                self.axes[3].scatter(post_spikes, [-40]*len(post_spikes),
                                     color="red", s=20, zorder=5)
            self.axes[3].set_ylim([-85, -35])
            self.axes[3].set_xlabel("Time (ms)", fontsize=8)

            self.fig.tight_layout(pad=1.5)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Simulation error", str(e))

# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = App(root)
    root.mainloop()