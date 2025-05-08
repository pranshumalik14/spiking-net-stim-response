import numpy as np
import matplotlib.pyplot as plt


class IzhikevichNetwork:
    def __init__(self,
                 n_exc: int,
                 n_inh: int,
                 n_input: int,
                 n_output: int,
                 dt: float = 1.0,
                 noise_std: float = 5.0,
                 seed: int = None,
                 A_LTP: float = None,
                 A_LTD: float = None,
                 tau_LTP: float = None,
                 tau_LTD: float = None):
        """
        n_exc, n_inh       : number of excitatory/inhibitory neurons
        n_input, n_output  : how many excitatory are input/output
        dt                 : simulation timestep in ms
        noise_std          : internal Gaussian noise SD (mV)
        """
        if seed is not None:
            np.random.seed(seed)

        # macro params
        self.N_exc = n_exc
        self.N_inh = n_inh
        self.N = n_exc + n_inh
        self.dt = dt
        self.noise_std = noise_std

        # neuron indices
        self.exc_idx = np.arange(n_exc)
        self.inh_idx = np.arange(n_exc, self.N)
        # randomly assign input/output tags (only among excitatory)
        perm = np.random.permutation(n_exc)
        self.input_idx = perm[:n_input]
        self.output_idx = perm[n_input:n_input+n_output]
        self.hidden_idx = np.setdiff1d(
            self.exc_idx,
            np.concatenate([self.input_idx, self.output_idx])
        )

        # Izhikevich parameters (a,b,c,d)
        self.a = np.zeros(self.N)
        self.b = np.zeros(self.N)
        self.c = np.zeros(self.N)
        self.d = np.zeros(self.N)
        # excitatory (regular-spiking)
        self.a[self.exc_idx] = 0.02
        self.b[self.exc_idx] = 0.2
        self.c[self.exc_idx] = -65.0
        self.d[self.exc_idx] = 8.0
        # inhibitory (fast-spiking)
        self.a[self.inh_idx] = 0.1
        self.b[self.inh_idx] = 0.2
        self.c[self.inh_idx] = -65.0
        self.d[self.inh_idx] = 2.0

        # neuron state variables
        self.v = np.full(self.N, -65.0)
        self.u = self.b * self.v

        # synaptic weights (wij: ni -> nj)
        self.W = np.zeros((self.N, self.N))
        self.W[self.exc_idx, :] = np.random.uniform(0, 5, (n_exc, self.N))
        self.W[self.inh_idx, :] = np.random.uniform(-5, 0, (n_inh, self.N))
        np.fill_diagonal(self.W, 0)
        # a copy of W (= Wc) is kept untouched to enable network resets
        self.Wc = np.copy(self.W)

        # short-term plasticity
        self.U = 0.2
        self.tau_d = 200.0
        self.tau_f = 600.0
        self.x = np.ones(self.N)
        self.u_stp = np.full(self.N, self.U)

        # spike-timing-dependent plasticity
        self.A_LTP = 1.0 if A_LTP is None else A_LTP
        self.A_LTD = np.random.uniform(0.8, 1.5) if A_LTD is None else A_LTD
        self.tau_LTP = 20.0 if tau_LTP is None else tau_LTP
        self.tau_LTD = np.random.uniform(
            20, 30) if tau_LTD is None else tau_LTD
        self.w_max = 20.0
        self.w_min = 0.0
        self.mu_decay = 5e-7
        self.last_spike = -np.inf * np.ones(self.N)

    def reset(self):
        self.W = np.copy(self.Wc)
        self.v.fill(-65.0)
        self.u = self.b * self.v
        self.x.fill(1.0)
        self.u_stp.fill(self.U)
        self.last_spike.fill(-np.inf)

    def _update_stp(self, fired):
        dx = ((1 - self.x) / self.tau_d -
              self.u_stp * self.x * fired)
        du = ((self.U - self.u_stp) / self.tau_f +
              self.U * (1 - self.u_stp) * fired)
        self.x += dx * self.dt
        self.u_stp += du * self.dt
        np.clip(self.u_stp, 0, 1, out=self.u_stp)
        np.clip(self.x, 0, 1, out=self.x)

    def _apply_stdp(self, t, fired):
        # only applied for E->E
        for n in np.where(fired)[0]:
            if n not in self.exc_idx:
                continue
            # note: last spike times are at most equal to `t` at this step
            delta_t = t - self.last_spike[self.exc_idx]  # Δt >= 0
            # first treating n as presynaptic neuron
            ns = self.exc_idx[delta_t > 0]  # indices of non-zero difference
            self.W[n, ns] -= (
                self.A_LTD * (1-1/self.tau_LTD)**delta_t[ns]
            )
            # then treating n as postsynaptic neuron
            self.W[ns, n] += (
                self.A_LTP * (1-1/self.tau_LTP)**delta_t[ns]
            )
        # finally applying weight decay and clipping to excitatory weights only
        exc_mat = np.ix_(self.exc_idx, self.exc_idx)
        self.W[exc_mat] *= (1 - self.mu_decay)
        self.W[exc_mat] = np.clip(self.W[exc_mat], self.w_min, self.w_max)

    def step(self, t: float, stim: np.ndarray = None):
        if stim is None:
            stim = np.zeros(self.N)
        fired = self.v >= 30.0
        self.v[fired] = self.c[fired]
        self.u[fired] += self.d[fired]
        self.last_spike[fired] = t
        s = self.u_stp * self.x  # short-term plasticity (stp)
        s[self.inh_idx] = 1.0  # only for outputs of excitatory neurons
        I_rec = self.W.T.dot(fired.astype(float) * s)  # apply stp in summation
        I_noise = np.random.randn(self.N) * self.noise_std
        I = I_rec + I_noise + stim
        dv1 = 0.04*self.v**2 + 5*self.v + 140 - self.u + I
        self.v += dv1 * (self.dt/2)
        dv2 = 0.04*self.v**2 + 5*self.v + 140 - self.u + I
        self.v += dv2 * (self.dt/2)
        du = self.a * (self.b*self.v - self.u)
        self.u += du * self.dt
        self._update_stp(fired.astype(float))
        self._apply_stdp(t, fired)
        return fired

    def run(self, T: float) -> np.ndarray:
        steps = int(T / self.dt)
        spikes = np.zeros((self.N, steps), dtype=bool)
        self.reset()
        for k in range(steps):
            t = k * self.dt
            spikes[:, k] = self.step(t)
        return spikes

    def plot_raster(self, spikes: np.ndarray, reaction_times: np.ndarray = None,
                    stim_times: np.ndarray = None):
        T = spikes.shape[1]
        times = np.arange(T) * self.dt/1000  # seconds
        plt.figure(figsize=(10, 6))
        groups = [
            (self.input_idx, 'Input', 'red'),
            (self.output_idx, 'Output', 'blue'),
            (self.hidden_idx, 'Hidden', 'gray'),
            (self.inh_idx, 'Inhibitory', 'black')
        ]
        boundaries = [0]
        for idxs, _, _ in groups:
            boundaries.append(boundaries[-1] + len(idxs))
        for i, (idxs, _, color) in enumerate(groups):
            start, end = boundaries[i], boundaries[i+1]
            plt.axhspan(start-0.5, end-0.5, facecolor=color, alpha=0.1)
            for j, neuron in enumerate(idxs, start=start):
                ts = times[spikes[neuron]]
                plt.scatter(ts, np.full_like(ts, j), s=2, c=color)
        for b in boundaries[1:-1]:
            plt.axhline(b-0.5, color='k', linewidth=0.5)
        mid = [(boundaries[i]+boundaries[i+1])/2 for i in range(len(groups))]
        labels = [name for _, name, _ in groups]
        plt.yticks(mid, labels, fontsize=10)
        if reaction_times is not None:
            # overlay reaction times as vertical lines in the output group region
            output_start = boundaries[1] - 0.5
            output_end = boundaries[2] - 0.5
            for rt in reaction_times:
                if not np.isnan(rt):
                    plt.vlines(rt/1000, output_start, output_end, color='blue',
                               linestyle='-', linewidth=2, alpha=0.5)
        if stim_times is not None:
            input_start = boundaries[0] - 0.5
            input_end = boundaries[1] - 0.5
            for stim in stim_times:
                plt.vlines(stim/1000, input_start, input_end, color='red',
                           linestyle='-', linewidth=2, alpha=0.5)
        plt.xlabel('Time (s)', fontsize=10)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    net = IzhikevichNetwork(80, 20, 10, 10, seed=42)
    spikes = net.run(1000)
    print(f"Dry run total spikes: {spikes.sum()}")
    net.plot_raster(spikes)
