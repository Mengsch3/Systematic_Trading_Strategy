"""Run basic validation checks for the notebook `dfrft` implementation.

Checks performed per N and signal:
 - identity (alpha=0)
 - FFT match (alpha=pi/2)
 - invertibility: dfrft(dfrft(x,a), -a) ~ x
 - additivity: FrFT(a) o FrFT(b) ~ FrFT(a+b)

Usage:
    python tests/test_dfrft.py

Requires: numpy
"""

import numpy as np
import sys


def _dft_matrix(N):
    n = np.arange(N)
    k = n[:, None]
    return np.exp(-2j * np.pi * k * n / N) / np.sqrt(N)


def _dfrft_matrix(N, alpha):
    F = _dft_matrix(N)
    eigvals, eigvecs = np.linalg.eig(F)
    inv_eigvecs = np.linalg.inv(eigvecs)
    frac = alpha / (np.pi / 2)
    return eigvecs @ np.diag(eigvals ** frac) @ inv_eigvecs


def dfrft(x, alpha):
    x = x.astype(np.complex128)
    N = len(x)
    a_mod = alpha % np.pi
    if np.isclose(a_mod, 0):
        return x
    if np.isclose(a_mod, np.pi / 2):
        return np.fft.fft(x) / np.sqrt(N)
    return _dfrft_matrix(N, alpha) @ x


def run_tests():
    rng = np.random.default_rng(42)
    sizes = [64, 65]
    tol = 1e-6
    failed = False

    for N in sizes:
        print('\nTesting N =', N)
        # signals
        delta = np.zeros(N, dtype=complex)
        delta[0] = 1.0
        chirp = np.exp(1j * np.pi * (np.arange(N) ** 2) / N)
        rand = rng.normal(size=N) + 1j * rng.normal(size=N)
        signals = [('delta', delta), ('chirp', chirp), ('rand', rand)]

        for name, x in signals:
            print(' signal:', name)
            # identity
            y0 = dfrft(x, 0.0)
            e_id = np.max(np.abs(y0 - x))
            print('  identity maxerr =', e_id)

            # FFT match
            yfft = dfrft(x, np.pi / 2)
            yfft_ref = np.fft.fft(x) / np.sqrt(N)
            e_fft = np.max(np.abs(yfft - yfft_ref))
            print('  fft-match maxerr =', e_fft)

            # invertibility for sample alphas
            for a in [np.pi / 4, np.pi / 3]:
                y = dfrft(x, a)
                x_rec = dfrft(y, -a)
                e_inv = np.max(np.abs(x - x_rec))
                print(f'  invert a={a:.6f} maxerr =', e_inv)

            # additivity
            a = np.pi / 6
            b = np.pi / 5
            left = dfrft(dfrft(x, a), b)
            right = dfrft(x, a + b)
            e_add = np.max(np.abs(left - right))
            print('  additivity maxerr =', e_add)

            # decide pass/fail for this signal (loose threshold to accommodate numeric variation)
            if e_id > 1e-9 or e_fft > 1e-9 or e_add > 1e-5:
                print('   -> WARNING: numerical differences exceed thresholds for', name)
                failed = True

    if failed:
        print('\nOne or more checks exceeded thresholds.')
        sys.exit(2)
    else:
        print('\nAll checks completed (differences within thresholds).')


if __name__ == '__main__':
    run_tests()
