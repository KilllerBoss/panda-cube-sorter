# toolchain/snn_py.py — exact numpy mirrors of the C++ perception pipeline.
# MUST stay numerically identical to native/src/{event_camera,alif_lsnn,pred_coder}.cpp
import numpy as np

EV_W, EV_H = 96, 72
BIN_COLS, BIN_ROWS = 12, 8
BINS_PER_CELL = 6
NUM_BINS = BIN_COLS * BIN_ROWS * BINS_PER_CELL  # 288


def rgb_to_luma_hue(rgb):
    """rgb: (H,W,3) uint8 -> luma (EV_H,EV_W), hue bucket (EV_H,EV_W) in {-1,0,1,2,3}"""
    r = rgb[:, :, 0].astype(np.float32) / 255.0
    g = rgb[:, :, 1].astype(np.float32) / 255.0
    b = rgb[:, :, 2].astype(np.float32) / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    L = 0.2126 * r + 0.7152 * g + 0.0722 * b
    hue = np.full(L.shape, -1.0, np.float32)
    sat = (mx - mn) > 0.18
    bright = mx > 0.15
    yellow = sat & bright & (mx == r) & (g > b + 0.12)
    red = sat & bright & (mx == r) & ~yellow
    green = sat & bright & (mx == g)
    blue = sat & bright & (mx == b)
    hue[yellow] = 3.0
    hue[red] = 0.0
    hue[green] = 1.0
    hue[blue] = 2.0
    # nearest resample to 96x72
    ys = (np.arange(EV_H) * (L.shape[0] / EV_H)).astype(int)
    xs = (np.arange(EV_W) * (L.shape[1] / EV_W)).astype(int)
    return L[np.ix_(ys, xs)], hue[np.ix_(ys, xs)]


class EventCamera:
    def __init__(self):
        self.luma = None
        self.hue = None

    def process(self, rgb, jitter_refresh, out_bins):
        L, H = rgb_to_luma_hue(rgb)
        out_bins[:] = 0.0
        n_events = 0
        if self.luma is None:
            self.luma, self.hue = L, H
            return 0
        dL = L - self.luma
        mask = (dL > 0.045) | (dL < -0.045)
        n_events = int(mask.sum())
        ys, xs = np.nonzero(mask)
        br = ys // (EV_H // BIN_ROWS)
        bc = xs // (EV_W // BIN_COLS)
        cell = br * BIN_COLS + bc
        pol = (dL[ys, xs] > 0).astype(np.int64)
        np.add.at(out_bins, cell * BINS_PER_CELL + pol, 1.0)
        hv = H[ys, xs]
        hmask = hv >= 0
        if np.any(hmask):
            np.add.at(out_bins, cell[hmask] * BINS_PER_CELL + 2 + hv[hmask].astype(np.int64), 1.0)
        self.luma, self.hue = L, H
        if jitter_refresh:
            out_bins *= 1.35
        out_bins *= 1.0 / ((EV_W // BIN_COLS) * (EV_H // BIN_ROWS))
        np.clip(out_bins, None, 2.0, out=out_bins)
        return n_events


class LsnnState:
    def __init__(self, n=128):
        self.v = np.zeros(n, np.float32)
        self.a = np.zeros(n, np.float32)
        self.spike = np.zeros(n, np.uint8)


class AlifLsnn:
    def __init__(self, n=128):
        self.n = n

    def step(self, bins, W, st):
        # W dict with lsnn_w (288x128), lsnn_b (128)
        st.v = W["lambda_v"] * st.v + W["lsnn_b"]
        act = bins > 0.001
        if np.any(act):
            st.v += bins[act] @ W["lsnn_w"][act]
        th = W["vth0"] * (1.0 + W["beta_th"] * st.a)
        spike = st.v > th
        st.v[spike] -= th[spike]
        st.spike[:] = spike.astype(np.uint8)
        st.a = W["rho_a"] * st.a + spike.astype(np.float32)
        return int(spike.sum())


class PredCoder:
    def __init__(self, n_bins=NUM_BINS):
        self.u_hat = np.zeros(n_bins, np.float32)

    def step(self, u, st, W, emb_out):
        # predict next input from adaptive state
        self.u_hat = st.a @ W["pred_fb"]  # (128,) @ (128,288)
        err = u - self.u_hat
        free_energy = 0.5 * float(err @ err)
        # local bounded Hebbian update
        if W["pred_eta"] > 0.0:
            g = (W["pred_eta"] * st.a / (1.0 + st.a * st.a))[:, None]
            upd = g * err[None, :]
            W["pred_fb"] = np.clip(W["pred_fb"] + upd, -0.5, 0.5)
        # projection [err ; a] -> emb
        cat = np.concatenate([err, st.a])
        emb_out[:] = np.maximum(cat @ W["emb_proj"], 0.0)
        return free_energy
