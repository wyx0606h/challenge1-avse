"""No-reference / text / speaker metrics for real-world AVSE evaluation.

Unlike :class:`look2hear.metrics.MetricsTracker` (which needs a clean
reference for SI-SDR / PESQ / STOI), the real-world test set has NO ground
truth, so this tracker aggregates:

    UTMOS     : UTMOSv2 no-reference MOS                       (higher better)
    DNSMOS    : P.835 p808 / sig / bak / ovr                   (higher better)
    CER       : Fun-ASR-Nano transcript vs human `text`        (lower better)
    spk_sim   : cosine(spk_emb(enhanced), enrollment_emb)      (higher better)

Heavy models are loaded lazily and only if their metric is requested, so a
smoke run with ``--metrics none`` (or a subset) never touches weights that are
not on disk. All loaders assume the weights were pre-fetched by
``download_models.sh`` and force offline mode.
"""

import os
import re
import csv

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Text normalization for Chinese CER/WER.
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(
    r"[\s,，。、？！；：“”‘’（）()\[\]【】《》〈〉…—\-~·.!?;:\"']+"
)


def normalize_zh(text):
    """Strip punctuation/whitespace so CER compares bare character sequences."""
    if text is None:
        return ""
    return _PUNCT_RE.sub("", str(text))


def _char_tokens(text):
    """Per-character token list for Chinese CER (one token per char)."""
    return list(normalize_zh(text))


def _edit_distance(ref, hyp):
    """Levenshtein distance between two token lists (Wagner-Fischer)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


# ---------------------------------------------------------------------------
# Metrics tracker.
# ---------------------------------------------------------------------------

ALL_METRICS = ("utmos", "dnsmos", "asr", "spk", "objective")


class RealMetricsTracker:
    """Aggregate no-reference metrics over the real-world test set.

    Args:
        metrics: which metrics to compute -- subset of
            ``{"utmos","dnsmos","asr","spk"}``. ``("none",)`` or ``[]``
            disables all (pipeline-only smoke run).
        device: torch device for UTMOS / WeSpeaker / DNSMOS.
        sample_rate: waveform sample rate (16 kHz).
        wespeaker_ckpt: path to WeSpeaker ``avg_model.pt`` (required for "spk").
        funasr_model_dir / funasr_remote_code: Fun-ASR-Nano model id + remote
            ``model.py`` (required for "asr").
        save_file: optional CSV path for per-item details + summary.
    """

    def __init__(self, metrics=ALL_METRICS, device="cuda", sample_rate=16000,
                 wespeaker_ckpt=None, funasr_model_dir="FunAudioLLM/Fun-ASR-Nano-2512",
                 funasr_remote_code=None, save_file=None):
        if metrics in (None, "none"):
            metrics = []
        self.metrics = set(m for m in metrics if m != "none")
        self.device = device
        self.sample_rate = sample_rate
        self.wespeaker_ckpt = wespeaker_ckpt
        self.funasr_model_dir = funasr_model_dir
        self.funasr_remote_code = funasr_remote_code
        self.save_file = save_file

        # Lazy-loaded models.
        self._utmos = None
        self._dnsmos_fn = None
        self._dnsmos_device_str = None  # resolved once on first DNSMOS call
        self._asr = None
        self._spk = None
        self._obj = None  # objective (reference-based) torchmetrics, remix only

        # Per-item rows and running accumulators.
        self.rows = []

    # ------------------------------------------------------------------ #
    # Lazy model loaders (offline).
    # ------------------------------------------------------------------ #

    def _get_utmos(self):
        if self._utmos is None:
            import utmosv2
            self._utmos = utmosv2.create_model(pretrained=True)
        return self._utmos

    def _get_dnsmos(self):
        if self._dnsmos_fn is None:
            from torchmetrics.functional.audio.dnsmos import (
                deep_noise_suppression_mean_opinion_score as fn,
            )
            self._dnsmos_fn = fn
        return self._dnsmos_fn

    def _get_asr(self):
        if self._asr is None:
            from funasr import AutoModel
            self._asr = AutoModel(
                model=self.funasr_model_dir,
                trust_remote_code=True,
                remote_code=self.funasr_remote_code,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
                hub="ms",
                disable_update=True,
                disable_pbar=True,   # silence FunASR's per-utterance rtf_avg bar
                disable_log=True,
            )
        return self._asr

    def _get_spk(self):
        if self._spk is None:
            from look2hear.models.wespeaker_resnet34 import WeSpeakerResNet34
            if not self.wespeaker_ckpt or not os.path.isfile(self.wespeaker_ckpt):
                raise FileNotFoundError(
                    f"WeSpeaker checkpoint not found: {self.wespeaker_ckpt}. "
                    "Run download_models.sh first."
                )
            self._spk = WeSpeakerResNet34(
                ckpt_path=self.wespeaker_ckpt, sample_rate=self.sample_rate
            ).eval().to(self.device)
        return self._spk

    def _get_obj(self):
        """Reference-based metrics (SI-SDR/PESQ/STOI), only used for remix GT."""
        if self._obj is None:
            from torchmetrics.audio import (
                ScaleInvariantSignalDistortionRatio,
                PerceptualEvaluationSpeechQuality,
                ShortTimeObjectiveIntelligibility,
            )
            self._obj = {
                "si_sdr": ScaleInvariantSignalDistortionRatio(),
                "pesq": PerceptualEvaluationSpeechQuality(self.sample_rate, "wb"),
                "stoi": ShortTimeObjectiveIntelligibility(self.sample_rate, False),
            }
        return self._obj

    # ------------------------------------------------------------------ #
    # Single-metric helpers (operate on a 1-D float waveform).
    # ------------------------------------------------------------------ #

    def _as_wave_1d(self, wav):
        """Coerce to a contiguous 1-D float32 tensor on CPU."""
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        wav = wav.detach().float().cpu()
        if wav.dim() > 1:
            wav = wav.reshape(-1)
        return wav.contiguous()

    def utmos_score(self, wav):
        model = self._get_utmos()
        wav = self._as_wave_1d(wav)
        out = model.predict(data=wav, sr=self.sample_rate, device=self.device,
                            verbose=False)
        return float(out)

    def _dnsmos_device(self):
        """Resolve the onnxruntime device string for DNSMOS, once.

        Returns an explicit ``cuda:N`` only if the GPU build of onnxruntime
        exposes CUDAExecutionProvider; otherwise ``cpu``. The index must be
        explicit -- torchmetrics passes ``device.index`` straight into
        ``OrtValue.ortvalue_from_numpy`` and ``provider_options``, and a bare
        ``"cuda"`` resolves to ``index=None`` which breaks that path. Each shard
        runs under its own CUDA_VISIBLE_DEVICES, so the visible GPU is always 0.

        Requires ``import torch`` to have run first (it has, at module load) so
        onnxruntime-gpu can borrow torch's bundled CUDA/cuDNN shared libraries.
        """
        if self._dnsmos_device_str is not None:
            return self._dnsmos_device_str
        dev = "cpu"
        if str(self.device).startswith("cuda"):
            try:
                import onnxruntime as ort
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    dev = "cuda:0"
                else:
                    print("[DNSMOS] CUDAExecutionProvider unavailable "
                          "(install onnxruntime-gpu); running on CPU.")
            except Exception as e:
                print(f"[DNSMOS] onnxruntime probe failed ({e}); running on CPU.")
        self._dnsmos_device_str = dev
        return dev

    def dnsmos_scores(self, wav):
        """Return dict {p808, sig, bak, ovr}.

        Runs on GPU when onnxruntime-gpu exposes CUDAExecutionProvider (see
        :meth:`_dnsmos_device`), else falls back to CPU. The waveform is fed as
        a CPU 1-D tensor; torchmetrics moves it onto the ORT device internally.
        """
        fn = self._get_dnsmos()
        wav = self._as_wave_1d(wav)  # CPU 1-D
        out = fn(wav, fs=self.sample_rate, personalized=False,
                 device=self._dnsmos_device())
        out = [float(x) for x in out]
        return {"dnsmos_p808": out[0], "dnsmos_sig": out[1],
                "dnsmos_bak": out[2], "dnsmos_ovr": out[3]}

    def transcribe(self, wav):
        """Run Fun-ASR-Nano on a waveform, return the recognized text."""
        asr = self._get_asr()
        wav = self._as_wave_1d(wav).numpy().astype(np.float32)
        res = asr.generate(input=[wav], cache={}, batch_size=1,
                          language="中文", itn=True, disable_pbar=True)
        return res[0]["text"] if res else ""

    def speaker_embedding(self, wav):
        spk = self._get_spk()
        wav = self._as_wave_1d(wav)
        return spk(wav)  # (256,) L2-normalized, on device

    def objective_scores(self, estimate, ref):
        """Reference-based SI-SDR / PESQ / STOI for one (estimate, ref) pair.

        Lengths are aligned to the shorter of the two. Each metric is wrapped
        so a degenerate clip (e.g. PESQ on silence) records None instead of
        aborting the run.
        """
        obj = self._get_obj()
        est = self._as_wave_1d(estimate)
        ref = self._as_wave_1d(ref)
        n = min(est.shape[-1], ref.shape[-1])
        est, ref = est[:n], ref[:n]
        out = {}
        for name, metric in obj.items():
            try:
                out[name] = float(metric(est, ref))
            except Exception as e:
                print(f"[{name}] objective failed: {e}")
        return out

    # ------------------------------------------------------------------ #
    # Per-item evaluation.
    # ------------------------------------------------------------------ #

    def __call__(self, estimate, meta, enroll_emb=None, ref=None):
        """Score one enhanced waveform.

        Args:
            estimate: enhanced waveform [T] (tensor/ndarray).
            meta: item dict from RealTestDataset (key/text/speaker_id/track/scene).
            enroll_emb: precomputed enrollment embedding (256,) for spk-sim, or
                None to skip the speaker metric for this item.
            ref: clean reference waveform [T] for objective metrics (remix only);
                None for mix items (no GT -> objective metrics skipped).

        Returns the per-item metric dict (also appended to self.rows).
        """
        row = {
            "key": meta.get("key", ""),
            "track": meta.get("track", ""),
            "scene": meta.get("scene", ""),
            "speaker_id": meta.get("speaker_id", ""),
        }

        if "utmos" in self.metrics:
            try:
                row["utmos"] = self.utmos_score(estimate)
            except Exception as e:
                print(f"[UTMOS] {row['key']}: {e}")

        if "dnsmos" in self.metrics:
            try:
                row.update(self.dnsmos_scores(estimate))
            except Exception as e:
                print(f"[DNSMOS] {row['key']}: {e}")

        if "asr" in self.metrics:
            try:
                hyp = self.transcribe(estimate)
                ref_text = meta.get("text", "")
                # CER: character-level edit distance over punctuation-stripped
                # text. Chinese has no word boundaries, so CER is the metric
                # (WER is not meaningful here).
                ref_tok = _char_tokens(ref_text)
                hyp_tok = _char_tokens(hyp)
                row["cer"] = _edit_distance(ref_tok, hyp_tok) / max(len(ref_tok), 1)
                row["hyp_text"] = hyp
                row["ref_text"] = ref_text
            except Exception as e:
                print(f"[ASR] {row['key']}: {e}")

        if "spk" in self.metrics and enroll_emb is not None:
            try:
                emb = self.speaker_embedding(estimate)
                enroll = enroll_emb.to(emb.device)
                row["spk_sim"] = float(torch.dot(emb, enroll) /
                                       (emb.norm() * enroll.norm() + 1e-8))
            except Exception as e:
                print(f"[SPK] {row['key']}: {e}")

        # Reference-based objective metrics: only for remix items (clean GT).
        if "objective" in self.metrics and ref is not None:
            try:
                row.update(self.objective_scores(estimate, ref))
            except Exception as e:
                print(f"[OBJECTIVE] {row['key']}: {e}")

        self.rows.append(row)
        return row

    # ------------------------------------------------------------------ #
    # Aggregation.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mean(rows, key):
        vals = []
        for r in rows:
            v = r.get(key, None)
            if v is None or v == "":
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        return float(np.mean(vals)) if vals else None

    # Numeric metric columns, in display/CSV order. Chinese -> CER only (no WER).
    NUMERIC_KEYS = ["si_sdr", "pesq", "stoi", "utmos",
                    "dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr",
                    "cer", "spk_sim"]

    def summarize(self):
        """Return nested summary: overall + per (track, scene) group means."""
        def group_summary(rows):
            means = {k: self._mean(rows, k) for k in self.NUMERIC_KEYS}
            return {k: v for k, v in means.items() if v is not None}

        summary = {"overall": group_summary(self.rows), "n": len(self.rows)}
        groups = {}
        for r in self.rows:
            g = f"{r.get('track','?')}/{r.get('scene','?')}"
            groups.setdefault(g, []).append(r)
        summary["groups"] = {g: {**group_summary(rs), "n": len(rs)}
                             for g, rs in sorted(groups.items())}
        return summary

    def final(self):
        summary = self.summarize()

        print("\n" + "=" * 60)
        print("Real-world AVSE Evaluation Results")
        print("=" * 60)
        print(f"Total items: {summary['n']}")
        print("\n-- Overall --")
        for k, v in summary["overall"].items():
            print(f"  {k:14s}: {v:.4f}")
        print("\n-- By track/scene --")
        for g, gs in summary["groups"].items():
            gs = dict(gs)          # copy: don't mutate the dict we return
            n = gs.pop("n", 0)
            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in gs.items())
            print(f"  {g:18s} (n={n}): {metric_str}")
        print("=" * 60)

        if self.save_file:
            self._save_csv()
        return summary

    def _save_csv(self):
        """Write three artifacts next to save_file:

            <save_file>                       -- per-item rows (all items)
            <stem>_<track>_<scene>.csv        -- per-group per-item rows
            <stem>_summary.csv                -- mean per metric, overall + groups
        """
        base = self.save_file
        stem, ext = os.path.splitext(base)
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

        lead = ["key", "track", "scene", "speaker_id"] + self.NUMERIC_KEYS + \
               ["ref_text", "hyp_text"]
        extra = [k for r in self.rows for k in r if k not in lead]
        cols = lead + sorted(set(extra))

        def write_rows(path, rows):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow(r)

        # 1. Total per-item table.
        write_rows(base, self.rows)
        print(f"Per-item results saved to {base}")

        # 2. Per-group per-item tables (track/scene).
        groups = {}
        for r in self.rows:
            g = (r.get("track", "?"), r.get("scene", "?"))
            groups.setdefault(g, []).append(r)
        for (track, scene), rows in sorted(groups.items()):
            gpath = f"{stem}_{track}_{scene}{ext}"
            write_rows(gpath, rows)
        print(f"Per-group tables: {stem}_<track>_<scene>{ext} "
              f"({len(groups)} groups)")

        # 3. Summary table: one row per scope (overall + each group), metric means.
        summary = self.summarize()
        summary_path = f"{stem}_summary{ext}"
        scols = ["scope", "n"] + self.NUMERIC_KEYS
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=scols, extrasaction="ignore")
            w.writeheader()
            w.writerow({"scope": "overall", "n": summary["n"],
                        **{k: round(v, 4) for k, v in summary["overall"].items()}})
            for g, gs in summary["groups"].items():
                gs = dict(gs)
                n = gs.pop("n", 0)
                w.writerow({"scope": g, "n": n,
                            **{k: round(v, 4) for k, v in gs.items()}})
        print(f"Summary saved to {summary_path}")

    @classmethod
    def merge_csvs(cls, shard_paths, out_csv):
        """Concatenate per-shard per-item CSVs and re-emit the 3 artifacts.

        Reuses ``_save_csv`` so the merged total / per-group / summary tables
        are identical in format to a single-process run. Missing shard files
        are skipped with a warning.
        """
        rows = []
        for p in shard_paths:
            if not os.path.isfile(p):
                print(f"[merge] WARN: missing shard {p}, skipped")
                continue
            with open(p, newline="") as f:
                rows.extend(list(csv.DictReader(f)))
        t = cls(metrics=[], save_file=out_csv)
        t.rows = rows
        t.final()
        return t
