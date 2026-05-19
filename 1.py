"""
Feature engineering for the energy instruments (cl1s, ho1s, rb1s, ng1s).
Notebook-style: only feature functions use `def`.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA


# =============================================================================
# CONFIG
# =============================================================================

INSTRUMENTS     = ['cl1s', 'ho1s', 'rb1s', 'ng1s']
UNSUP_TRAIN_END = '2019-12-31'
RANDOM_STATE    = 42
SP_WINDOW       = 64       # rolling window length for spectral features


# =============================================================================
# F1. RAW OHLCV + OPEN INTEREST FEATURES
# =============================================================================

def feat_returns(df):
    """Log returns and simple transforms."""
    out = pd.DataFrame(index=df.index)
    log_close = np.log(df['close'])
    out['logret_1']   = log_close.diff()
    out['logret_abs'] = out['logret_1'].abs()
    out['logret_sq']  = out['logret_1'] ** 2
    return out


def feat_volatility(df, window=20):
    """Close-to-close vol, Parkinson vol, vol-of-vol."""
    out = pd.DataFrame(index=df.index)
    logret = np.log(df['close']).diff()
    out[f'vol_cc_{window}']   = logret.rolling(window).std()
    out[f'vov_{window}']      = out[f'vol_cc_{window}'].rolling(window).std()
    hl = np.log(df['high'] / df['low'])
    park_var = (hl ** 2) / (4.0 * np.log(2.0))
    out[f'vol_park_{window}'] = np.sqrt(park_var.rolling(window).mean())
    return out


def feat_range(df, window=14):
    """Intraday range and ATR (normalised by close)."""
    out = pd.DataFrame(index=df.index)
    out['range_norm'] = (df['high'] - df['low']) / df['close']
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low']  - prev_close).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out[f'atr_{window}'] = tr.rolling(window).mean() / df['close']
    return out


def feat_volume(df, window=20):
    """Log-volume and rolling z-score."""
    out = pd.DataFrame(index=df.index)
    log_vol = np.log(df['volume'].clip(lower=1))
    mu = log_vol.rolling(window).mean()
    sd = log_vol.rolling(window).std()
    out['log_volume'] = log_vol
    out['vol_z']      = (log_vol - mu) / (sd + 1e-9)
    return out


def feat_open_interest(df, window=20):
    """OI z-score and signed OI change."""
    out = pd.DataFrame(index=df.index)
    log_oi = np.log(df['open_interest'].clip(lower=1))
    mu = log_oi.rolling(window).mean()
    sd = log_oi.rolling(window).std()
    out['oi_z']             = (log_oi - mu) / (sd + 1e-9)
    out['oi_signed_change'] = df['open_interest'].diff() * np.sign(df['close'].diff())
    return out


# =============================================================================
# F2. FORMULAIC ALPHAS (typical price replaces VWAP)
# =============================================================================

def feat_alphas(df):
    out = pd.DataFrame(index=df.index)
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    logret = np.log(df['close']).diff()
    out['a1']  = (np.sign(logret) * logret.abs().rolling(5).std()).rolling(5).mean()
    out['a6']  = -df['close'].rolling(10).corr(df['volume'])
    out['a41'] = (tp - df['close']) / df['close']
    hh = df['high'].rolling(9).max()
    ll = df['low'].rolling(9).min()
    out['a53'] = (df['close'] - ll) / (hh - ll + 1e-9)
    out['a54'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
    ret5 = df['close'].pct_change(5)
    out['a84'] = np.sign(ret5) * ret5.abs()
    ma20 = df['close'].rolling(20).mean()
    sd20 = df['close'].rolling(20).std()
    out['a32'] = (df['close'] - ma20) / (sd20 + 1e-9)
    return out


# =============================================================================
# F3. CROSS-SECTIONAL ENERGY FEATURES
# =============================================================================

def feat_cross_sectional(instrument, returns_panel, vol_panel):
    """Where this instrument sits relative to the rest of the complex."""
    out = pd.DataFrame(index=returns_panel.index)
    ret_ranks = returns_panel.rank(axis=1, pct=True)
    vol_ranks = vol_panel.rank(axis=1, pct=True)
    out['xs_ret_rank']       = ret_ranks[instrument]
    out['xs_vol_rank']       = vol_ranks[instrument]
    out['xs_dispersion']     = returns_panel.std(axis=1)
    composite                = returns_panel.mean(axis=1)
    out['xs_excess_ret']     = returns_panel[instrument] - composite
    out['xs_corr_composite'] = returns_panel[instrument].rolling(60).corr(composite)
    return out


# =============================================================================
# F4. SIGNAL PROCESSING FEATURES (DFrFT, S-transform, Hilbert, SSA)
# =============================================================================

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


def dfrft(x, alpha, matrix=None):
    """
    Discrete Fractional Fourier Transform via fractional power of the unitary DFT matrix.
    alpha = 0 -> identity, alpha = pi/2 -> standard FFT.
    """
    N = len(x)
    a_mod = alpha % np.pi
    if np.isclose(a_mod, 0):
        return x.astype(complex)
    if np.isclose(a_mod, np.pi / 2):
        return np.fft.fft(x) / np.sqrt(N)
    if matrix is not None:
        return matrix @ x.astype(complex)
    return _dfrft_matrix(N, alpha) @ x.astype(complex)


def hilbert_transform(x):
    """Analytic signal via FFT (multiply positive freqs by 2, kill negatives)."""
    N = len(x)
    X = np.fft.fft(x)
    H = np.zeros(N, dtype=complex)
    if N % 2 == 0:
        H[0]      = X[0]
        H[N // 2] = X[N // 2]
        H[1:N // 2] = 2 * X[1:N // 2]
    else:
        H[0] = X[0]
        H[1:(N + 1) // 2] = 2 * X[1:(N + 1) // 2]
    return np.fft.ifft(H)


def ssa_singular_values(x):
    """Singular Spectrum Analysis: SVD of the trajectory (Hankel) matrix."""
    N = len(x)
    L = N // 2
    K = N - L + 1
    H = np.zeros((L, K))
    for i in range(L):
        H[i] = x[i:i + K]
    s = np.linalg.svd(H, compute_uv=False)
    return s


def feat_signal_processing(df, window=SP_WINDOW):
    """
    Rolling spectral features on log-returns.
    Per window we extract:
      - DFrFT at alpha = pi/4: low-band E, high-band E, spectral entropy, centroid
      - S-transform proxy (FFT in low/mid/high bands + entropy)
      - Hilbert analytic signal: mean/std of amplitude and instantaneous frequency
      - SSA: trend / cycle / noise energy fractions
    """
    logret = np.log(df['close']).diff().fillna(0.0).values
    n = len(logret)

    # output buffers
    dfrft_low      = np.full(n, np.nan)
    dfrft_high     = np.full(n, np.nan)
    dfrft_entropy  = np.full(n, np.nan)
    dfrft_centroid = np.full(n, np.nan)
    dfrft_alpha_star    = np.full(n, np.nan)   # NEW: optimal alpha
    dfrft_entropy_star  = np.full(n, np.nan)   # NEW: entropy at the optimum

    s_low     = np.full(n, np.nan)
    s_mid     = np.full(n, np.nan)
    s_high    = np.full(n, np.nan)
    s_entropy = np.full(n, np.nan)

    h_amp_mean  = np.full(n, np.nan)
    h_amp_std   = np.full(n, np.nan)
    h_freq_mean = np.full(n, np.nan)
    h_freq_std  = np.full(n, np.nan)

    ssa_trend = np.full(n, np.nan)
    ssa_cycle = np.full(n, np.nan)
    ssa_noise = np.full(n, np.nan)

    alpha_val = 0.5 * np.pi / 2   # half-rotation - between time and frequency
    alpha_grid = np.arange(0.05, 1.0, 0.1) * (np.pi / 2)
    alpha_list = np.concatenate([alpha_grid, [alpha_val]])
    dfrft_matrices = {a: _dfrft_matrix(window, a) for a in alpha_list}
    L4 = window // 4
    L3 = window // 3
    freqs = np.arange(window)

    for t in range(window - 1, n):
        x = logret[t - window + 1 : t + 1]

        # --- DFrFT ---
        entropies  = np.zeros(len(alpha_grid))

        for i, a in enumerate(alpha_grid):
            Y_a   = dfrft_matrices[a] @ x
            mag_a = np.abs(Y_a) ** 2
            p_a   = mag_a / (mag_a.sum() + 1e-12)
            entropies[i] = -(p_a * np.log(p_a + 1e-12)).sum()

        # alpha* = the alpha that minimises entropy (most concentrated transform)
        star_idx = int(np.argmin(entropies))
        dfrft_alpha_star[t]   = alpha_grid[star_idx] / (np.pi / 2)   # normalise to [0,1]
        dfrft_entropy_star[t] = entropies[star_idx]

        # keep the 4 "default" features at alpha = pi/4 for consistency
        Y     = dfrft_matrices[alpha_val] @ x
        mag   = np.abs(Y) ** 2
        total = mag.sum() + 1e-12
        p     = mag / total
        dfrft_low[t]      = mag[:L4].sum() / total
        dfrft_high[t]     = mag[-L4:].sum() / total
        dfrft_entropy[t]  = -(p * np.log(p + 1e-12)).sum()
        dfrft_centroid[t] = (freqs * p).sum()

        # --- S-transform proxy (FFT energy in 3 bands + entropy) ---
        X     = np.fft.fft(x)
        pwr   = np.abs(X) ** 2
        total = pwr.sum() + 1e-12
        p     = pwr / total
        s_low[t]     = pwr[:L3].sum() / total
        s_mid[t]     = pwr[L3:2 * L3].sum() / total
        s_high[t]    = pwr[2 * L3:].sum() / total
        s_entropy[t] = -(p * np.log(p + 1e-12)).sum()

        # --- Hilbert ---
        analytic = hilbert_transform(x)
        amp      = np.abs(analytic)
        phase    = np.unwrap(np.angle(analytic))
        ifreq    = np.diff(phase) / (2 * np.pi)
        h_amp_mean[t]  = amp.mean()
        h_amp_std[t]   = amp.std()
        h_freq_mean[t] = ifreq.mean()
        h_freq_std[t]  = ifreq.std()

        # --- SSA ---
        s     = ssa_singular_values(x)
        s2    = s ** 2
        total = s2.sum() + 1e-12
        ssa_trend[t] = s2[0] / total
        ssa_cycle[t] = s2[1:3].sum() / total
        ssa_noise[t] = s2[3:].sum() / total

    out = pd.DataFrame(index=df.index)
    out['dfrft_low']      = dfrft_low
    out['dfrft_high']     = dfrft_high
    out['dfrft_entropy']  = dfrft_entropy
    out['dfrft_centroid'] = dfrft_centroid
    out['s_low']          = s_low
    out['s_mid']          = s_mid
    out['s_high']         = s_high
    out['s_entropy']      = s_entropy
    out['h_amp_mean']     = h_amp_mean
    out['h_amp_std']      = h_amp_std
    out['h_freq_mean']    = h_freq_mean
    out['h_freq_std']     = h_freq_std
    out['ssa_trend']      = ssa_trend
    out['ssa_cycle']      = ssa_cycle
    out['ssa_noise']      = ssa_noise
    out['dfrft_alpha_star']   = dfrft_alpha_star
    out['dfrft_entropy_star'] = dfrft_entropy_star
    return out


# =============================================================================
# F5. GMM REGIME PROBABILITIES
# =============================================================================

def fit_gmm(X_train, n_components=3, random_state=RANDOM_STATE):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    gmm = GaussianMixture(
        n_components=n_components, covariance_type='full',
        n_init=5, max_iter=200, random_state=random_state,
    )
    gmm.fit(X_train_scaled)
    return scaler, gmm


def gmm_features(X, scaler, gmm):
    X_scaled = scaler.transform(X)
    probs = gmm.predict_proba(X_scaled)
    cols = ['gmm_p' + str(k) for k in range(gmm.n_components)]
    return pd.DataFrame(probs, index=X.index, columns=cols)


# =============================================================================
# F6. HMM FILTERING PROBABILITIES (FORWARD ALGORITHM FROM LECTURE)
# =============================================================================

def fit_hmm(X_train, n_states=3, random_state=RANDOM_STATE):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    hmm = GaussianHMM(
        n_components=n_states, covariance_type='full',
        n_iter=200, random_state=random_state,
    )
    hmm.fit(X_train_scaled)
    return scaler, hmm


def hmm_forward_filter(X, scaler, hmm):
    """
    Filtering probabilities  p(H_t = k | X_1, ..., X_t).
    Forward recursion from the lecture:
        alpha_1 = Gamma(1) * pi
        alpha_t = Gamma(t) * Q^T * alpha_{t-1}
        xi_t    = alpha_t / sum(alpha_t)
    Causal - uses only past observations (no smoothing).
    """
    X_scaled = scaler.transform(X)
    n_obs = X_scaled.shape[0]
    K = hmm.n_components

    # emission probabilities (with small floor to avoid zeros)
    emission = np.zeros((n_obs, K))
    for k in range(K):
        emission[:, k] = multivariate_normal.pdf(
            X_scaled, mean=hmm.means_[k], cov=hmm.covars_[k], allow_singular=True
        )
    emission = np.clip(emission, 1e-300, None)

    # work in log-space to avoid underflow in forward recursion
    def logsumexp(a, axis=None):
        a = np.asarray(a)
        max_a = np.max(a, axis=axis, keepdims=True)
        res = max_a + np.log(np.sum(np.exp(a - max_a), axis=axis, keepdims=True))
        return np.squeeze(res, axis=axis)

    log_emission = np.log(emission)
    log_pi = np.log(hmm.startprob_ + 1e-300)
    log_A = np.log(hmm.transmat_ + 1e-300)

    log_alpha = np.zeros((n_obs, K))
    log_alpha[0] = log_pi + log_emission[0]
    for t in range(1, n_obs):
        # for each current state k, sum over previous states j: log_alpha_prev[j] + log_A[j,k]
        s = log_alpha[t - 1][:, None] + log_A
        log_alpha[t] = logsumexp(s, axis=0) + log_emission[t]

    # normalize to get filtering probabilities
    log_norm = logsumexp(log_alpha, axis=1)
    alpha = np.exp(log_alpha - log_norm[:, None])

    cols = ['hmm_p' + str(k) for k in range(K)]
    return pd.DataFrame(alpha, index=X.index, columns=cols)


# =============================================================================
# F7. K-MEANS DISTANCE FEATURES
# =============================================================================

def fit_kmeans(X_train, n_clusters=4, random_state=RANDOM_STATE):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    km.fit(X_train_scaled)
    return scaler, km


def kmeans_features(X, scaler, km):
    X_scaled = scaler.transform(X)
    distances = km.transform(X_scaled)
    cols = ['km_dist_' + str(k) for k in range(km.n_clusters)]
    return pd.DataFrame(distances, index=X.index, columns=cols)

# =============================================================================
# F8. PCA ON ROLLING RETURN WINDOWS
# =============================================================================

def build_return_windows(df, window=SP_WINDOW):
    """
    Build a matrix where each row t is the last `window` log returns ending at t.
    The first window-1 rows are NaN (window not yet full).
    """
    logret = np.log(df['close']).diff().fillna(0.0).values
    n = len(logret)
    W = np.full((n, window), np.nan)
    for t in range(window - 1, n):
        W[t] = logret[t - window + 1 : t + 1]
    return pd.DataFrame(W, index=df.index)


def fit_pca(W_train, n_components=3, random_state=RANDOM_STATE):
    """Fit PCA on a matrix of training windows."""
    scaler = StandardScaler()
    W_train_scaled = scaler.fit_transform(W_train)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(W_train_scaled)
    return scaler, pca


def pca_features(W, scaler, pca):
    """
    Project each window onto the top PCs and compute reconstruction error.
    Returns n_components + 1 columns.
    """
    W_scaled = scaler.transform(W)
    proj     = pca.transform(W_scaled)
    recon    = pca.inverse_transform(proj)
    err      = np.sqrt(((W_scaled - recon) ** 2).sum(axis=1))
    cols_pc  = ['pca_pc' + str(i) for i in range(pca.n_components_)]
    out      = pd.DataFrame(proj, index=W.index, columns=cols_pc)
    out['pca_recon_err'] = err
    return out

# =============================================================================
# === LOAD OHLCV ==============================================================
# =============================================================================

def load_ohlcv(path='ohlcv_data.csv'):
    ohlcv = pd.read_csv(path)
    ohlcv['date'] = pd.to_datetime(ohlcv['date'])
    ohlcv = ohlcv.sort_values(['instrument', 'date']).reset_index(drop=True)
    return ohlcv

def build_ohlcv_dict(ohlcv):
    ohlcv_dict = {}
    for inst in INSTRUMENTS:
        sub = ohlcv[ohlcv['instrument'] == inst].copy()
        sub = sub.set_index('date').sort_index()
        sub = sub[['open', 'high', 'low', 'close', 'volume', 'open_interest']]
        ohlcv_dict[inst] = sub
    return ohlcv_dict

def build_feature_dictionary(ohlcv_path='ohlcv_data.csv'):
    ohlcv = load_ohlcv(ohlcv_path)
    ohlcv_dict = build_ohlcv_dict(ohlcv)

    returns_panel = pd.concat(
        [np.log(ohlcv_dict[i]['close']).diff().rename(i) for i in INSTRUMENTS], axis=1
    )
    vol_panel = pd.concat(
        [np.log(ohlcv_dict[i]['close']).diff().rolling(20).std().rename(i) for i in INSTRUMENTS],
        axis=1,
    )

    features_dict = {}

    for inst in INSTRUMENTS:
        print('\nBuilding features for', inst, '...')
        df = ohlcv_dict[inst]

        # ---- F1 + F2 : stateless per-instrument features ----
        base = pd.concat([
            feat_returns(df),
            feat_volatility(df, window=20),
            feat_range(df, window=14),
            feat_volume(df, window=20),
            feat_open_interest(df, window=20),
            feat_alphas(df),
        ], axis=1)

        # ---- F3 : cross-sectional ----
        xs = feat_cross_sectional(inst, returns_panel, vol_panel).reindex(base.index)

        # ---- F4 : signal-processing features (rolling, slow) ----
        print('  - signal processing (rolling window =', SP_WINDOW, ')')
        sp = feat_signal_processing(df, window=SP_WINDOW)

        # ---- input vector for GMM / HMM / KMeans ----
        unsup_input = base[['logret_1', 'vol_cc_20', 'range_norm', 'vol_z', 'oi_z']].copy()
        train_mask  = (unsup_input.index <= UNSUP_TRAIN_END) & unsup_input.notna().all(axis=1)
        score_mask  = unsup_input.notna().all(axis=1)
        X_train     = unsup_input.loc[train_mask].values
        X_score     = unsup_input.loc[score_mask]

        # ---- F5 : GMM ----
        print('  - GMM')
        gmm_scaler, gmm = fit_gmm(X_train, n_components=3)
        gmm_df = gmm_features(X_score, gmm_scaler, gmm).reindex(base.index)

        # ---- F6 : HMM with forward filtering ----
        print('  - HMM (forward filter)')
        hmm_scaler, hmm = fit_hmm(X_train, n_states=3)
        hmm_df = hmm_forward_filter(X_score, hmm_scaler, hmm).reindex(base.index)

        # ---- F7 : KMeans ----
        print('  - KMeans')
        km_scaler, km = fit_kmeans(X_train, n_clusters=4)
        km_df = kmeans_features(X_score, km_scaler, km).reindex(base.index)

        # ---- F8 : PCA on rolling return windows ----
        print('  - PCA on rolling return windows')
        W_full          = build_return_windows(df, window=SP_WINDOW)
        train_mask_pca  = (W_full.index <= UNSUP_TRAIN_END) & W_full.notna().all(axis=1)
        score_mask_pca  = W_full.notna().all(axis=1)
        W_train         = W_full.loc[train_mask_pca].values
        W_score         = W_full.loc[score_mask_pca]
        pca_scaler, pca = fit_pca(W_train, n_components=3)
        pca_df          = pca_features(W_score, pca_scaler, pca).reindex(base.index)

        # ---- combine ----
        all_features = pd.concat([base, xs, sp, gmm_df, hmm_df, km_df, pca_df], axis=1)
        all_features['instrument'] = inst
        all_features.index.name = 'date'
        features_dict[inst] = all_features

    return features_dict

if __name__ == '__main__':
    features_dict = build_feature_dictionary()

    # =============================================================================
    # === SANITY CHECK ============================================================
    # =============================================================================

    for inst, df_feat in features_dict.items():
        print('\n' + inst, ':', df_feat.shape)
        print('Columns:', list(df_feat.columns))
        print(df_feat.tail(3))

    for inst, df in features_dict.items():
        df.to_csv(f'features_{inst}.csv')

