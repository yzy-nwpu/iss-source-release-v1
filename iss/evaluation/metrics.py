"""Task metrics used by the evaluation recipes: SI-SNR, BSS Eval SDR, PESQ nb."""
from __future__ import annotations

import itertools


def measure(reference, estimate, sample_rate: int = 8000) -> dict:
    import numpy as np
    import torch
    from mir_eval.separation import bss_eval_sources
    from pesq import pesq
    from speechbrain.nnet.losses import get_si_snr_with_pitwrapper

    reference = np.asarray(reference, dtype=np.float32)
    estimate = np.asarray(estimate, dtype=np.float32)
    if reference.shape != estimate.shape or reference.ndim != 2:
        raise ValueError("Reference and estimate must both have shape [time, speakers]")
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
        raise ValueError("Metrics require finite waveforms")
    si_snr = -float(get_si_snr_with_pitwrapper(torch.from_numpy(reference[None]),
                                               torch.from_numpy(estimate[None])).mean())
    sdr, _sir, _sar, sdr_permutation = bss_eval_sources(reference.T, estimate.T,
                                                        compute_permutation=True)
    pesq_candidates, pesq_errors = [], []
    for permutation in itertools.permutations(range(reference.shape[1])):
        try:
            score = float(np.mean([pesq(sample_rate, reference[:, index],
                                        estimate[:, permutation[index]], "nb")
                                   for index in range(reference.shape[1])]))
            pesq_candidates.append((score, list(permutation)))
        except Exception as error:
            pesq_errors.append({"permutation": list(permutation),
                                "error": f"{type(error).__name__}: {error}"})
    if not pesq_candidates:
        pesq_value, pesq_permutation = None, None
    else:
        pesq_value, pesq_permutation = max(pesq_candidates)
    return {"si_snr": si_snr, "sdr": float(np.mean(sdr)),
            "sdr_permutation": list(sdr_permutation), "pesq": pesq_value,
            "pesq_permutation": pesq_permutation, "pesq_errors": pesq_errors}
