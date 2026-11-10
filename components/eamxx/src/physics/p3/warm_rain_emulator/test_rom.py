"""Offline check of rom_warm_rain.

  1. P3's update equations (IPARAM = 3), warm_rain_emulator_merge's clipping and
     nc_conservation, applied to the slot rates, reproduce the ROM's tendencies,
     except where P3 caps a number loss at what the cell holds (reported, and checked
     to end at zero);
  2. the masks fall back on P3 exactly where the module's criteria say;
  3. fallback cells keep the ROM's rates (for warm_rain_emulator_replaces_p3: true);
  4. the module scripts to TorchScript (for the LibTorch backend).

    python test_rom_warm_rain.py /path/to/model.pt
"""
import sys
import warnings

import numpy as np
import torch

import rom_warm_rain as R

model = sys.argv[1]
emu = R.create_emulator({'model_path': model, 'num_threads': '1'})
m = emu.model
dt = m.target_period_s
print(f"model {model}: period {dt:g} s;  fallback: q_liq <= {m.q_liq_min:g} kg/kg, "
      f"r_mean <= {m.r_mean_min * 1e6:g} um, no drops, or cloud tail > {m.cloud_tail_max:g} x nr")
# What the model file has frozen in (older exports may lack some of these)
attrs = []
for a in ('n_bins', 'r_split', 'dt', 'limit_nr', 'use_dopri5', 'rtol', 'n_substeps', 'qsmall'):
    if hasattr(m.rom, a):
        v = getattr(m.rom, a)
        attrs.append(f"{a}={v * 1e6:.2f}um" if a == 'r_split' else f"{a}={v}")
print("  model settings:", ", ".join(attrs))

# Synthetic (ncol, nlev) state
rng = np.random.default_rng(0)
ncol, nlev = 64, 72
lu = lambda lo, hi: 10 ** rng.uniform(np.log10(lo), np.log10(hi), (ncol, nlev))
rho = rng.uniform(0.6, 1.2, (ncol, nlev))
qc, nc, qr, nr = lu(1e-7, 2e-3), lu(1e7, 1e9), lu(1e-7, 1e-3), lu(1e2, 1e6)
qc[:8] = 0.0                                  # rain only
qr[8:16] = 0.0                                # cloud only
qc[16:20] = qr[16:20] = 0.0                   # no liquid
nc[20, :36] = 0.0; qr[20, :36] = 0.0          # cloud mass without drops
q_bdry = R.Q_LIQ_MIN                          # exactly at the liquid threshold (strict >)
qc[21, :], qr[21, :] = q_bdry, 0.0
inputs = {k: np.ascontiguousarray(v) for k, v in zip(R.INPUTS, (qc, nc, qr, nr, rho))}
for v in inputs.values():
    v.setflags(write=False)                   # P3 hands over read-only arrays
outputs = {k: np.full((ncol, nlev), np.nan) for k in R.OUTPUTS}
emu.infer(inputs, outputs)
o = outputs
assert all(np.all(np.isfinite(v)) for v in o.values()), "non-finite outputs"

# --- 1. P3's update reproduces the ROM ------------------------------------
au = np.maximum(o['qc2qr_autoconv_tend'], 0); ac = np.maximum(o['qc2qr_accret_tend'], 0)
ncautr = np.maximum(o['ncautr'], 0)
nc_sc = np.maximum(np.minimum(o['nc_selfcollect_tend'], 0), -np.maximum(nc, 0) / dt)
nr_sc = np.minimum(np.maximum(o['nr_selfcollect_tend'], 0), np.maximum(nr, 0) / dt + ncautr)
nc_acc = np.maximum(o['nc_accret_tend'], 0); nc2nr = np.maximum(o['nc2nr_autoconv_tend'], 0)
# nc_conservation: the cloud-number sinks are scaled to what the cell holds
sink_nc = (nc_acc + nc2nr) * dt
source_nc = nc + nc_sc * dt
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(sink_nc > source_nc, source_nc / sink_nc, 1.0)
nc_acc_l, nc2nr_l = nc_acc * ratio, nc2nr * ratio
d_qc = -(ac + au); d_qr = ac + au
d_nc = -nc_acc_l - nc2nr_l + nc_sc
d_nr = ncautr - nr_sc

f = lambda a: torch.from_numpy(np.ascontiguousarray(a))
mu_c = 0.0005714 * (f(nc) * 1e-6 * f(rho)) + 0.2714
mu_c = torch.clamp(1.0 / mu_c**2 - 1.0, 2.0, 15.0)
with torch.no_grad():
    T = m.rom(f(nc * rho), f(nr * rho), f(qc * rho), f(qr * rho), mu_c, torch.full_like(mu_c, R.MU_R))
dNr, dNc, dqc, dqr = [t.numpy() / rho for t in T]
# Where the merge caps a number loss at what the cell holds
nc_clip = (nc_sc != o['nc_selfcollect_tend']) | (ratio < 1)
nr_clip = nr_sc != o['nr_selfcollect_tend']
no_clip = np.zeros_like(nc_clip)
for name, got, want, clip in [('qc', d_qc, dqc, no_clip), ('qr', d_qr, dqr, no_clip),
                              ('nc', d_nc, dNc, nc_clip), ('nr', d_nr, dNr, nr_clip)]:
    scale = max(np.abs(want).max(), 1e-300)
    err = np.abs(got - want)[~clip].max() / scale
    print(f"  d{name}/dt through P3's slots vs ROM: max err / max|ROM| = {err:.1e}"
          + (f"  (outside {clip.sum()} capped cells)" if clip.any() else ""))
    assert err < 1e-12, name

# In capped cells the ROM asks to remove more drops than the cell holds: the merge
# removes exactly all of them
use_rom = o['use_cloud'] == 1
for name, n, d, want, clip in [('Nc', nc, d_nc, dNc, nc_clip), ('Nr', nr, d_nr, dNr, nr_clip)]:
    if not clip.any():
        continue
    after = n + d * dt
    assert np.all(np.abs(after[clip]) <= 1e-12 * n[clip]), f"{name}: capped cells must end at zero"
    ratio = -want[clip] * dt / n[clip]
    kinds = {'rain only': (qc <= 0) & (qr > 0), 'cloud only': (qc > 0) & (qr <= 0), 'both': (qc > 0) & (qr > 0)}
    where = ", ".join(f"{k} {np.sum(clip & v)}" for k, v in kinds.items() if np.any(clip & v))
    print(f"  {name}: in {clip.sum()} cells the ROM removes {ratio.min():.2f} to {ratio.max():.2f} times "
          f"the drops held; capped to all of them ({where}; {np.sum(clip & use_rom)} where the ROM applies, "
          f"{np.sum(clip & ~use_rom)} where P3 takes over)")
assert all(np.all(o[k][16:20] == 0) for k in R.OUTPUTS), "cells without liquid must be zero"
assert all(np.all(o[k][:8] == 0) for k in ('qc2qr_autoconv_tend', 'ncautr', 'nc_selfcollect_tend')), \
    "rain-only cells must have no cloud rates"

# --- 2. masks ----------------------------------------------------------------
liquid = (qc > 0) | (qr > 0)
q_liq = np.where(qc > 0, qc, 0) + np.where(qr > 0, qr, 0)
n_liq = np.where(qc > 0, nc, 0) + np.where(qr > 0, nr, 0)
r_mean = np.cbrt(3 * q_liq / (4 * np.pi * R.RHOW * np.where(n_liq > 0, n_liq, 1)))
no_drops = liquid & (n_liq <= 0)
low_q = liquid & ~no_drops & (q_liq <= R.Q_LIQ_MIN)
small = liquid & ~no_drops & ~low_q & (r_mean <= R.R_MEAN_MIN)
# cloud drops above the split radius, from the ROM's own binning of the cloud gamma
with torch.no_grad():
    has_c = f(qc).reshape(-1) > 0
    dn_c, _ = m.rom._bin_gamma(f(nc * rho).reshape(-1), f(qc * rho).reshape(-1), mu_c.reshape(-1), has_c)
    tail_bins = ((dn_c * (1 - m.rom.cloud_mask)).sum(-1) * m.rom.dlnr).numpy().reshape(qc.shape) / rho
# (the bins stop at 5 mm: the tail above it is lost there, negligible against the criterion)
tail = liquid & ~no_drops & ~low_q & ~small & (qr > 0) & (tail_bins > m.cloud_tail_max * nr * (1 + 1e-9))
borderline = liquid & ~no_drops & ~low_q & ~small & (qr > 0) & \
    (np.abs(tail_bins - m.cloud_tail_max * nr) <= 1e-9 * nr)
expect = liquid & ~no_drops & ~low_q & ~small & ~tail
assert set(np.unique(o['use_cloud'])) <= {0.0, 1.0}, "masks must be 0 or 1"
assert np.array_equal(o['use_rain'], o['use_cloud']), "use_rain must equal use_cloud"
assert np.array_equal((o['use_cloud'] == 1)[~borderline], expect[~borderline]), "masks differ from the criteria"
assert np.all(o['use_cloud'][21] == 0), "liquid exactly at Q_LIQ_MIN must fall back (strict >)"
assert np.all(o['use_cloud'][20, :36] == 0), "mass without drops must fall back"
print(f"  of {liquid.sum()} cells with liquid: {no_drops.sum()} no drops, {low_q.sum()} too little liquid, "
      f"{small.sum()} drops too small, {tail.sum()} cloud tail outnumbers rain -> ROM in {expect.sum()}")

# --- 3. fallback cells keep the ROM's rates ----------------------------------
fb = liquid & ~expect & (qc > 0)
assert np.any(o['qc2qr_autoconv_tend'][fb] > 0), "fallback cells must keep the ROM's rates"

# --- 4. TorchScript -------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    scripted = torch.jit.script(m)
args = [f(inputs[n]) for n in R.INPUTS]
with torch.no_grad():
    diff = max(float((a - b).abs().max()) / max(float(a.abs().max()), 1e-300)
               for a, b in zip(m(*args), scripted(*args)))
# (TorchScript re-optimizes a module after its first calls, which can change round-off)
print(f"  scripted module vs eager: max rel diff {diff:.1e}")
assert diff < 1e-12
print("PASS")