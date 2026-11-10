#!/usr/bin/env python3
"""Check the synthetic cases of test p3_warm_rain_emulator against P3 alone.

  check_synthetic.py CASE CASE_FILE STOCK_FILE

  no_rain: with an emulator of zero rates that applies everywhere, P3 has no
           autoconversion and no accretion, where P3 alone has some.
  drain:   with an emulator that converts the cloud water it is given to rain,
           there is no cloud water left after a step, where P3 alone has some.
  rom:     with the AE-SINDy ROM, P3 has no accretion (the ROM reports all of its
           mass conversion as autoconversion), and has autoconversion.
  rom_fallback: with the AE-SINDy ROM, but P3's own rates in the cells where the
           ROM falls back: P3 has autoconversion, and accretion only from P3.
"""
import sys

import netCDF4
import numpy as np

case = sys.argv[1]
emu, stock = [netCDF4.Dataset(f) for f in sys.argv[2:4]]


def values(ds, name):
    # (the first record is the initial condition)
    return np.asarray(ds[name][1:], dtype=np.float64)


ok = True
names = dict(no_rain=('qc2qr_autoconv', 'qc2qr_accret'), drain=('qc',),
             rom=('qc2qr_accret',), rom_fallback=())[case]
for name in names:
    a, s = values(emu, name), values(stock, name)
    print(f'{name}: max with the emulator {np.abs(a).max():.3e}, max in P3 alone {np.abs(s).max():.3e}')
    ok = ok and np.all(a == 0) and np.abs(s).max() > 0
if case in ('rom', 'rom_fallback'):
    # The ROM ran, and its rates reached P3's update
    au = values(emu, 'qc2qr_autoconv')
    print(f'qc2qr_autoconv: max with the ROM {np.abs(au).max():.3e}')
    if case == 'rom_fallback':
        # (P3's own accretion, in the cells where the ROM falls back)
        print(f"qc2qr_accret: max with the fallback {np.abs(values(emu, 'qc2qr_accret')).max():.3e}")
    ok = ok and np.abs(au).max() > 0 and all(np.all(np.isfinite(values(emu, v))) for v in ('qc', 'nc', 'qr', 'nr'))
sys.exit(0 if ok else 1)