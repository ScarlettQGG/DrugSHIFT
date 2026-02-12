def normalize_modalities_inplace(
    modalities_dict,
    zscore=True,
    l2=True,
    eps=1e-8,
    skip_names=None,
):
    skip_names = set(skip_names or [])
    for m, pdict in modalities_dict.items():
        if m in skip_names:
            continue
        if not pdict:
            continue
        X = np.stack(list(pdict.values()), axis=0).astype(np.float32)
        if zscore:
            mu = X.mean(axis=0, keepdims=True)
            sd = np.maximum(X.std(axis=0, keepdims=True), eps)
            X = (X - mu) / sd
        if l2:
            norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
            X = X / norms
        for i, k in enumerate(pdict.keys()):
            pdict[k] = X[i]

def normalize_secms_inplace(
    modalities_dict: dict[str, dict[str, np.ndarray]],
    secms_names: list[str],
    *,
    log1p: bool = True,
    smoothing: bool = True,
    smooth_window: int = 3,   # odd is best; 3/5/7
    per_protein_area: bool = True,   # sum-to-1
    per_protein_center: bool = False, # subtract mean across fractions (shape only)
    per_protein_scale: bool = False,  # divide by std across fractions (shape only)
    clip_quantile: float | None = 0.999,  # winsorize extreme spikes per modality
    l2: bool = True,
    eps: float = 1e-8,
) -> None:
    """
    SEC-MS specific normalization (rows=proteins, cols=fractions).

    Recommended defaults:
      - log1p=True
      - smoothing=True (small window)
      - per_protein_area=True
      - per_protein_center/scale=False (enable only if you want pure shape)
      - l2=True

    Modifies modalities_dict in-place for secms_names only.
    """

    def _smooth_1d(x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return x
        w = int(w)
        if w % 2 == 0:
            w += 1
        pad = w // 2
        # reflect padding to avoid edge artifacts
        xp = np.pad(x, (pad, pad), mode="reflect")
        kernel = np.ones(w, dtype=np.float32) / float(w)
        return np.convolve(xp, kernel, mode="valid").astype(np.float32)

    for m in secms_names:
        if m not in modalities_dict:
            continue
        pdict = modalities_dict[m]
        if not pdict:
            continue

        # stack to [N, F]
        keys = list(pdict.keys())
        X = np.stack([np.asarray(pdict[k], dtype=np.float32) for k in keys], axis=0)

        # SEC-MS is typically non-negative; if yours can be negative (e.g., already centered), skip/adjust log.
        if log1p:
            # make safe: shift minimally if any negatives exist
            minv = float(X.min())
            if minv < 0:
                X = X - minv
            X = np.log1p(X)

        # optional winsorize to suppress rare spike artifacts (per-modality)
        if clip_quantile is not None:
            q = float(clip_quantile)
            q = min(max(q, 0.5), 1.0)
            hi = np.quantile(X, q)
            X = np.clip(X, 0.0, hi).astype(np.float32)

        # optional smoothing along fractions (per protein)
        if smoothing and smooth_window and smooth_window > 1:
            w = int(smooth_window)
            X = np.stack([_smooth_1d(X[i], w) for i in range(X.shape[0])], axis=0)

        # per-protein area normalization: remove abundance scale, keep elution shape
        if per_protein_area:
            s = X.sum(axis=1, keepdims=True)
            s = np.maximum(s, eps)
            X = X / s

        # optional center/scale across fractions within each protein (pure shape)
        if per_protein_center:
            X = X - X.mean(axis=1, keepdims=True)
        if per_protein_scale:
            sd = X.std(axis=1, keepdims=True)
            sd = np.maximum(sd, eps)
            X = X / sd

        # final L2 to make cosine behavior comparable to other modalities
        if l2:
            n = np.linalg.norm(X, axis=1, keepdims=True)
            n = np.maximum(n, eps)
            X = X / n

        # write back
        for i, k in enumerate(keys):
            pdict[k] = X[i].astype(np.float32)
