"""AE-SINDy collision-coalescence ROM, for P3's warm-rain emulator.

The model file is a TorchScript ROM written by export_rom.py
(ekdejong/cpp_SCREAM_python_ROM_interface). It holds the encoder, the SINDy
dynamics, the decoder, the integrator, the cloud/rain split radius and the
integration period dt. A retrained or re-exported ROM is a file swap.

Used with warm_rain_emulator_python_module: rom_warm_rain, and
warm_rain_emulator_model_path: the model file.

Contract (see eamxx_p3_warm_rain_emulator.cpp); arrays are (ncol, nlev) doubles:
  inputs : qc, nc, qr, nr [kg/kg, #/kg; cell averages, dry], rho [kg/m3, dry]
  outputs: qc2qr_autoconv_tend, qc2qr_accret_tend, ncautr, nc2nr_autoconv_tend,
           nc_accret_tend, nc_selfcollect_tend, nr_selfcollect_tend (cell averages,
           with P3's signs), and use_cloud, use_rain

The ROM gives the total tendencies of collision-coalescence (dqc, dNc, dqr, dNr),
not P3's separate processes. They go into P3's slots so that P3's update
(IPARAM = 3, SCREAM's default) reproduces them:
  qc2qr_autoconv_tend = -dqc/dt            qc2qr_accret_tend = 0
  nc_selfcollect_tend =  dNc/dt  (<= 0)    nc2nr_autoconv_tend = nc_accret_tend = 0
  ncautr              = max(dNr/dt, 0)     nr_selfcollect_tend = max(-dNr/dt, 0)
The cloud-number loss goes in nc_selfcollect_tend, not nc2nr_autoconv_tend, because
nr_conservation counts nc2nr_autoconv_tend as a source of rain number. It needs
warm_rain_emulator_cloud_self_collection: true (the default). P3's diagnostics
report all of the mass conversion as autoconversion.

The rates are means over the dt frozen into the model file, not over P3's step
(the emulator is not given it): export the ROM with --dt equal to P3's step.

Fallback on P3 (with warm_rain_emulator_replaces_p3: false): use_cloud = use_rain = 0,
so P3 uses its own rates, where the liquid (qc + qr) is at most Q_LIQ_MIN, or its
mean-volume radius is at most R_MEAN_MIN, or it has no drops, or there is rain and
more cloud drops than rain drops already lie above the ROM's cloud/rain split radius
(the ROM counts those as rain, so its rain-number tendency there is a small
difference of large numbers). The rates themselves
are the ROM's in every cell with liquid, so that with warm_rain_emulator_replaces_p3:
true (masks ignored) the ROM is used everywhere.
"""
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

INPUTS = ('qc', 'nc', 'qr', 'nr', 'rho')
OUTPUTS = ('qc2qr_autoconv_tend', 'qc2qr_accret_tend', 'ncautr', 'nc2nr_autoconv_tend',
           'nc_accret_tend', 'nc_selfcollect_tend', 'nr_selfcollect_tend', 'use_cloud', 'use_rain')

# Rain gamma shape given to the ROM: keep equal to p3::constant_mu_rain
MU_R = 1.0
# Fall back on P3 for cells with only small liquid water or tiny droplets
R_MEAN_MIN = 3e-6   # mean-volume radius of the liquid [m]
Q_LIQ_MIN = 1e-5    # qc + qr [kg/kg of dry air]
RHOW = 1000.0       # Reference density of water [kg/m3]
# Fall back on P3 where, with rain present, the cloud drops above the ROM's split
# radius outnumber the rain drops by more than this factor
CLOUD_TAIL_MAX = 1.0


class RomWarmRain(nn.Module):
    """P3's warm-rain slot rates from the ROM's collision-coalescence tendencies."""

    def __init__(self, model_file: str):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)  # torch.jit deprecation notice
            self.rom = torch.jit.load(model_file, map_location='cpu')
        self.rom.eval()
        self.mu_r = MU_R
        self.q_liq_min = Q_LIQ_MIN
        self.r_mean_min = R_MEAN_MIN
        self.rhow = RHOW
        self.cloud_tail_max = CLOUD_TAIL_MAX
        self.d_split = 2.0 * float(self.rom.r_split)  # split diameter [m]
        # The rates are means over this period [s]: the model is meant for P3 steps of this length
        self.target_period_s = float(self.rom.dt)

    def forward(self, qc: torch.Tensor, nc: torch.Tensor, qr: torch.Tensor, nr: torch.Tensor,
                rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                            torch.Tensor]:
        qc, nc, qr, nr, rho = [v.double() for v in (qc, nc, qr, nr, rho)]

        # Cloud gamma shape as in P3's get_cloud_dsd2 (Martin et al. 1994), nc in cm^-3
        mu_c = 0.0005714 * (nc * 1e-6 * rho) + 0.2714
        mu_c = torch.clamp(1.0 / (mu_c * mu_c) - 1.0, 2.0, 15.0)

        # The ROM works per volume of air, and skips cells without liquid itself
        dNr, dNc, dqc, _ = self.rom(nc * rho, nr * rho, qc * rho, qr * rho,
                                    mu_c, torch.full_like(qc, self.mu_r))
        inv_rho = 1.0 / rho
        dqc, dNc, dNr = dqc * inv_rho, dNc * inv_rho, dNr * inv_rho  # per kg of dry air

        zero = torch.zeros_like(qc)
        # Where there is sufficient liquid and large enough droplets the ROM replaces P3
        has_c, has_r = qc > 0, qr > 0
        q_liq = torch.where(has_c, qc, zero) + torch.where(has_r, qr, zero)   # [kg/kg]
        n_liq = torch.where(has_c, nc, zero) + torch.where(has_r, nr, zero)   # [#/kg]
        has_drops = n_liq > 0
        n_safe = torch.where(has_drops, n_liq, torch.ones_like(n_liq))      # avoid division by zero
        r_mean = torch.pow(3.0 * q_liq / (4.0 * 3.141593 * self.rhow * n_safe), 1.0 / 3.0)

        # Cloud drops above the split radius (counted as rain by the ROM) [#/kg]:
        # Nc * Q(mu_c + 1, lambda_c * D_split), Q the regularized upper incomplete gamma
        nc_safe = torch.where(has_c & (nc > 0), nc, torch.ones_like(nc))
        qc_safe = torch.where(has_c & (nc > 0), qc, torch.ones_like(qc))
        lam_c = torch.pow(3.141593 / 6.0 * self.rhow * nc_safe
                          * (mu_c + 1.0) * (mu_c + 2.0) * (mu_c + 3.0) / qc_safe, 1.0 / 3.0)
        cloud_tail = torch.where(has_c & (nc > 0),
                                 nc * torch.special.gammaincc(mu_c + 1.0, lam_c * self.d_split), zero)
        tail_ok = ~has_r | (cloud_tail <= self.cloud_tail_max * nr)

        rom_ok = (has_c | has_r) & has_drops & (q_liq > self.q_liq_min) & (r_mean > self.r_mean_min) & tail_ok
        use = rom_ok.double()
        return (torch.clamp(-dqc, min=0.0),   # qc2qr_autoconv_tend
                zero,                         # qc2qr_accret_tend
                torch.clamp(dNr, min=0.0),    # ncautr
                zero,                         # nc2nr_autoconv_tend
                zero,                         # nc_accret_tend
                torch.clamp(dNc, max=0.0),    # nc_selfcollect_tend
                torch.clamp(-dNr, min=0.0),   # nr_selfcollect_tend
                use,                          # use_cloud
                use)                          # use_rain


class PythonBackendEmulator:
    """What the python backend of components/emulators calls: infer(inputs, outputs)."""

    def __init__(self, model_file: str):
        self.model = RomWarmRain(model_file)

    def infer(self, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray]):
        with warnings.catch_warnings():
            # (the inputs are read-only arrays, and are only read)
            warnings.filterwarnings('ignore', message='The given NumPy array is not writable')
            args = [torch.from_numpy(inputs[n]) for n in INPUTS]
        with torch.no_grad():
            res = self.model(*args)
        for name, v in zip(OUTPUTS, res):
            outputs[name][...] = v.numpy()


def create_emulator(config):
    # Each MPI rank uses its own threads only
    torch.set_num_threads(int(config.get('num_threads', 1)))
    return PythonBackendEmulator(config['model_path'])