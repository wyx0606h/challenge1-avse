"""Real-world AVSE evaluation over the REAL-AVSE corpus.

Runs a trained AV-ConvTasNet checkpoint over the REAL-AVSE test tree and reports
metrics per target speaker (s1 and s2 of every clip):

    mix   (real two-talker, NO clean GT): UTMOS, DNSMOS, WER/CER, speaker sim
    remix (synthetic, has s1/s2.wav GT) : the above + SI-SDR / PESQ / STOI

Data is read straight from the corpus directories (no preprocess/*.json):

    REAL-AVSE/track{1,2}/{dev,test}/{mix,remix}/<id>/{mix.wav, s{1,2}.{mp4,txt,wav}}

Speaker similarity uses per-speaker enrollment voiceprints built from the clean
single-speaker sources (remix s1.wav/s2.wav), grouped by speaker label.

All metric models must be pre-downloaded by download_models.sh; this script runs
fully offline (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set by run_eval_real.sh).

Example:
    python eval_real.py --track both --scene both --split test --metrics all \
        --save_dir enhanced_out
"""

import os
import glob
import argparse
import warnings

import torch
import yaml
from tqdm import tqdm

import look2hear.models
from look2hear.datas.real_test_dataset import (
    RealTestDataset, build_items, build_enrollment_items, REAL_AVSE_ROOT,
)
from look2hear.metrics.real_metrics import RealMetricsTracker, ALL_METRICS

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Real-world AVSE evaluation")
    p.add_argument("--conf_dir",
                   default="Experiments/Track1-AVConvTasNet-Baseline/conf.yml",
                   help="Config yaml (for audionet_config to rebuild the model)")
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint source: a local Lightning *.ckpt, a local "
                        "serialize best_model.pth, OR a HuggingFace repo id "
                        "(e.g. JusperLee/Real-World-AVSE-Baseline-Track1) whose "
                        "config.json + model.safetensors are downloaded on first "
                        "use. Default: latest epoch=*.ckpt next to conf.")
    p.add_argument("--data_root", default=REAL_AVSE_ROOT,
                   help="REAL-AVSE corpus root (holds track1/ and track2/)")
    p.add_argument("--track", default="both", choices=["track1", "track2", "both"])
    p.add_argument("--scene", default="both", choices=["mix", "remix", "both"],
                   help="mix=real no-GT, remix=synthetic with clean GT")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--metrics", default="all",
                   help="Comma list of {utmos,dnsmos,asr,spk,objective}, 'all', 'none'")
    p.add_argument("--wespeaker_ckpt",
                   default="pretrained/wespeaker_cnceleb_resnet34/model_5.pt",
                   help="WeSpeaker checkpoint for speaker similarity")
    p.add_argument("--enroll_ckpt", default=None,
                   help="Precomputed enrollment voiceprints (.pt: {speaker_id: emb}). "
                        "If given and exists, load it instead of rebuilding from "
                        "clean remix sources. Build one with build_enrollment.py.")
    p.add_argument("--funasr_model", default="FunAudioLLM/Fun-ASR-Nano-2512")
    p.add_argument("--funasr_remote_code", default="Fun-ASR/model.py")
    p.add_argument("--mode", default="both",
                   choices=["enhance", "eval", "both"],
                   help="enhance=only run model + save wavs (no metrics); "
                        "eval=only score pre-saved wavs in --save_dir (no model); "
                        "both=enhance + score in one pass (default)")
    p.add_argument("--save_dir", default=None,
                   help="Dir for enhanced wavs (mirrors corpus layout). "
                        "Required for --mode enhance/eval.")
    p.add_argument("--out_csv", default=None,
                   help="Per-item CSV path. Default: eval_results/<split>/real_metrics.csv")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Split the item list into N shards for multi-GPU parallel runs")
    p.add_argument("--shard_id", type=int, default=0,
                   help="Which shard this process handles (0..num_shards-1)")
    p.add_argument("--merge_shards", action="store_true",
                   help="Merge existing per-shard CSVs into the final tables and exit "
                        "(no model/metrics run). Use after all shards finish.")
    p.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap manifest entries per (track,scene) (smoke test)")
    p.add_argument("--no_align_face", action="store_true",
                   help="Disable landmark-based face alignment; use the legacy "
                        "whole-face resize (for A/B comparison)")
    return p.parse_args()


def resolve_metrics(arg):
    if arg.strip().lower() == "all":
        return set(ALL_METRICS)
    if arg.strip().lower() == "none":
        return set()
    return set(m.strip() for m in arg.split(",") if m.strip())


def find_latest_ckpt(conf_dir):
    """Pick last.ckpt, else the newest epoch=*.ckpt next to the config."""
    exp_dir = os.path.dirname(os.path.abspath(conf_dir))
    last = os.path.join(exp_dir, "last.ckpt")
    if os.path.isfile(last):
        return last
    cands = glob.glob(os.path.join(exp_dir, "*.ckpt"))
    if not cands:
        raise FileNotFoundError(f"No checkpoint found in {exp_dir}")
    return max(cands, key=os.path.getmtime)


def is_hf_repo_id(s):
    """True if ``s`` looks like a HuggingFace ``org/name`` repo id (not a local
    path and not a checkpoint filename), e.g.
    ``"JusperLee/Real-World-AVSE-Baseline-Track1"``."""
    import re
    return (
        isinstance(s, str)
        and not os.path.exists(s)
        and re.fullmatch(r"[\w.-]+/[\w.-]+", s) is not None
        and not s.endswith((".ckpt", ".pth", ".pt", ".tar"))
    )


def load_model_hf_or_serialize(ckpt, conf, device):
    """Load the model from a HuggingFace repo id OR a local serialize() payload.

    * HF repo id (``org/name``) -> ``from_pretrained`` (the PyTorchModelHubMixin
      path): downloads ``config.json`` + ``model.safetensors`` and rebuilds the
      exact architecture. Auth-aware, so private repos use the cached HF token.
    * local ``best_model.pth`` -> legacy ``from_pretrain``: a self-describing
      ``{model_name, state_dict, model_args}`` payload.

    Either way the trained video-encoder weights are inside the checkpoint, so no
    external lip-reading backbone is needed (``video_pretrain`` is not in the HF
    config; the local serialize path forces it to ``None``).
    """
    model_cls = getattr(look2hear.models, conf["audionet"]["audionet_name"])
    if is_hf_repo_id(ckpt):
        model = model_cls.from_pretrained(ckpt)        # mixin: config + safetensors
        print(f"Loaded HF:{ckpt} via from_pretrained (config.json + safetensors)")
    else:
        model = model_cls.from_pretrain(ckpt, video_pretrain=None)   # local payload
        print(f"Loaded {os.path.basename(ckpt)} via from_pretrain (serialize format)")
    return model.eval().to(device)


def load_model(conf, ckpt_path, device):
    """Rebuild AV_ConvTasNet from config and load Lightning ckpt weights.

    The training checkpoint stores weights under ``audio_model.*`` (Lightning
    system prefix) with no ``model_args``, so we rebuild from ``audionet_config``
    (video_pretrain=None -- the trained video encoder weights live in the ckpt)
    and strip the prefix, mirroring train.py:load_model_weights.
    """
    acfg = conf["audionet"]["audionet_config"]
    video_cfg = conf.get("videonet", {}).get("videonet_config", {})
    model = getattr(look2hear.models, conf["audionet"]["audionet_name"])(
        sample_rate=conf["datamodule"]["data_config"]["sample_rate"],
        video_relu_type=video_cfg.get("relu_type", "prelu"),
        video_pretrain=None,
        **acfg,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_sd = state["state_dict"] if "state_dict" in state else state
    model_sd = {k[len("audio_model."):]: v for k, v in full_sd.items()
                if k.startswith("audio_model.")}
    if not model_sd:
        model_sd = full_sd
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    real_missing = [m for m in missing if "num_batches_tracked" not in m]
    print(f"Loaded {os.path.basename(ckpt_path)} | missing={len(real_missing)} "
          f"unexpected={len(unexpected)}")
    return model.eval().to(device)


def build_enrollment(tracker, data_root, tracks, split, device, limit=None,
                     cache_path=None, save_path=None):
    """Per-speaker enrollment embeddings, from a cache or freshly computed.

    Sources are the remix scene's s1.wav/s2.wav (clean), grouped by speaker;
    a speaker's voiceprint is the mean of its clean-source embeddings.
    Returns ``enroll[speaker_id] = mean_embedding`` (256-d).

    Args:
        cache_path: if given and exists, load the precomputed dict and skip the
            (slow) WeSpeaker extraction over thousands of clean clips.
        save_path: if given, write the freshly built dict here for reuse /
            release (e.g. the dev voiceprints shipped with the baseline).
    """
    import soundfile as sf

    if cache_path and os.path.isfile(cache_path):
        enroll = torch.load(cache_path, map_location="cpu")
        print(f"Loaded enrollment voiceprints from {cache_path} "
              f"({len(enroll)} speakers).")
        return {k: v.to(device) for k, v in enroll.items()}

    print("Building speaker enrollment voiceprints from clean remix sources...")
    pairs = build_enrollment_items(root=data_root, tracks=tracks, split=split,
                                   limit=limit)
    per_speaker = {}  # speaker_id -> list of emb
    seen_paths = set()
    for spk, wav_path in tqdm(pairs, desc="enroll"):
        # A clean source can appear in many remixes; embed each unique wav once.
        if wav_path in seen_paths:
            continue
        seen_paths.add(wav_path)
        wav = sf.read(wav_path, dtype="float32")[0]
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        emb = tracker.speaker_embedding(torch.from_numpy(wav)).detach()
        per_speaker.setdefault(spk, []).append(emb)

    enroll = {spk: torch.stack(embs).mean(dim=0)
              for spk, embs in per_speaker.items()}
    print(f"Enrollment ready for {len(enroll)} speakers "
          f"({len(seen_paths)} clean clips).")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({k: v.cpu() for k, v in enroll.items()}, save_path)
        print(f"Saved enrollment voiceprints to {save_path}")
    return {k: v.to(device) for k, v in enroll.items()}


def enhanced_save_path(save_dir, meta):
    """Mirror the corpus layout under save_dir.

    <save_dir>/<track>/<split>/<scene>/<clip_id>/s{1,2}.wav
    """
    return os.path.join(save_dir, meta["track"], meta["split"], meta["scene"],
                        meta["clip_id"], f"{meta['spk_tag']}.wav")


def shard_csv_path(out_csv, num_shards, shard_id):
    """Per-shard CSV name; the single-shard case keeps the plain name."""
    if num_shards <= 1:
        return out_csv
    stem, ext = os.path.splitext(out_csv)
    return f"{stem}.shard{shard_id}of{num_shards}{ext}"


def main():
    args = parse_args()

    # Output CSVs go to a standalone eval_results/ folder (NOT the experiment
    # dir), keyed by split, so the folder name never collides with the data
    # track. Override the exact path with --out_csv.
    out_csv = args.out_csv or os.path.join("eval_results", args.split,
                                           "real_metrics.csv")

    # Merge-only: combine per-shard CSVs into the final tables and exit.
    if args.merge_shards:
        shard_paths = [shard_csv_path(out_csv, args.num_shards, i)
                       for i in range(args.num_shards)]
        print(f"Merging {len(shard_paths)} shard CSVs -> {out_csv}")
        RealMetricsTracker.merge_csvs(shard_paths, out_csv)
        print("Done.")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode in ("enhance", "eval") and not args.save_dir:
        raise ValueError(f"--mode {args.mode} requires --save_dir")

    need_model = args.mode in ("enhance", "both")
    need_metrics = args.mode in ("eval", "both")
    metrics = resolve_metrics(args.metrics) if need_metrics else set()

    print("=" * 60)
    print("Real-world AVSE Evaluation")
    print(f"  mode    : {args.mode}")
    print(f"  metrics : {sorted(metrics) if metrics else '(none)'}")
    print(f"  device  : {device}")
    if args.num_shards > 1:
        print(f"  shard   : {args.shard_id} / {args.num_shards}")
    print("=" * 60)

    with open(args.conf_dir) as f:
        conf = yaml.safe_load(f)
    face_size = conf["datamodule"]["data_config"].get("face_size", 96)
    sr = conf["datamodule"]["data_config"]["sample_rate"]

    # 1. Model (skipped in eval-only mode).
    model = None
    if need_model:
        # Three checkpoint sources, auto-detected from --ckpt:
        #   * HF repo id ("org/name")        -> from_pretrained (config + safetensors)
        #   * local serialize best_model.pth -> from_pretrain (self-describing)
        #   * local Lightning *.ckpt / none  -> rebuild from conf.yml + strip prefix
        ckpt = args.ckpt or find_latest_ckpt(args.conf_dir)
        if is_hf_repo_id(ckpt) or (ckpt.endswith(".pth") and os.path.isfile(ckpt)):
            model = load_model_hf_or_serialize(ckpt, conf, device)
        else:
            model = load_model(conf, ckpt, device)

    # 2. Data (read straight from the corpus tree), then shard.
    tracks = ("track1", "track2") if args.track == "both" else (args.track,)
    scenes = ("mix", "remix") if args.scene == "both" else (args.scene,)
    items = build_items(root=args.data_root, tracks=tracks, split=args.split,
                        scenes=scenes, limit=args.limit)
    if args.num_shards > 1:
        items = items[args.shard_id::args.num_shards]
        print(f"Shard {args.shard_id}: {len(items)} of the items")
    dataset = RealTestDataset(items, face_size=face_size, sample_rate=sr,
                              align_face=not args.no_align_face)

    # 3. Metrics tracker (only when scoring).
    tracker = None
    if need_metrics:
        csv_path = shard_csv_path(out_csv, args.num_shards, args.shard_id)
        tracker = RealMetricsTracker(
            metrics=metrics, device=device, sample_rate=sr,
            wespeaker_ckpt=args.wespeaker_ckpt,
            funasr_model_dir=args.funasr_model,
            funasr_remote_code=args.funasr_remote_code,
            save_file=csv_path,
        )

    # 4. Speaker enrollment (only if speaker metric requested).
    enroll = None
    if "spk" in metrics:
        enroll = build_enrollment(tracker, args.data_root, tracks, args.split,
                                  device, limit=args.limit,
                                  cache_path=args.enroll_ckpt)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # 5. Main loop.
    import soundfile as sf
    import torchaudio
    desc = {"enhance": "enhance", "eval": "score", "both": "eval"}[args.mode]
    _warned_no_enroll = set()
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=desc):
            meta = dataset.items[idx]
            save_path = (enhanced_save_path(args.save_dir, meta)
                         if args.save_dir else None)

            # Obtain the enhanced waveform: run the model, or load a saved wav.
            if need_model:
                # A single corrupt mp4/wav must not abort a multi-thousand-item
                # run -- skip it (consistent with the eval-only missing-wav skip).
                try:
                    mix, mouth, meta = dataset[idx]
                    mix = mix.to(device)
                    mouth = mouth.to(device)
                    if mouth.ndim == 3:
                        mouth = mouth.unsqueeze(0)     # [1, Tv, H, W]
                    enhanced = model(mix.unsqueeze(0), mouth).squeeze(0).cpu()  # [T]
                except Exception as e:
                    print(f"[{args.mode}] skipped {meta.get('key', idx)}: {e}")
                    continue
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    # Write 32-bit float PCM so a later --mode eval reads back the
                    # exact same samples (16-bit PCM would quantize + clip the
                    # un-normalized model output, making both != enhance+eval).
                    torchaudio.save(save_path, enhanced.unsqueeze(0), sr,
                                    encoding="PCM_F", bits_per_sample=32)
            else:  # eval-only: read the pre-saved enhanced wav
                if not os.path.isfile(save_path):
                    print(f"[eval] missing enhanced wav, skipped: {save_path}")
                    continue
                enhanced = torch.from_numpy(sf.read(save_path, dtype="float32")[0])

            if not need_metrics:
                continue

            enroll_emb = (enroll.get(meta["speaker_id"])
                          if enroll is not None else None)
            if enroll is not None and enroll_emb is None and "spk" in metrics:
                # Speaker has no enrollment voiceprint -> spk_sim silently
                # skipped for this item; warn once so coverage gaps are visible.
                if meta["speaker_id"] not in _warned_no_enroll:
                    _warned_no_enroll.add(meta["speaker_id"])
                    print(f"[spk] no enrollment for speaker '{meta['speaker_id']}'"
                          f" -> spk_sim skipped for its items")
            # Clean reference for objective metrics (remix only).
            ref = None
            if "objective" in metrics and meta.get("ref_path"):
                ref = torch.from_numpy(sf.read(meta["ref_path"], dtype="float32")[0])

            tracker(enhanced, meta, enroll_emb=enroll_emb, ref=ref)

    if tracker is not None:
        tracker.final()
    print("Done.")


if __name__ == "__main__":
    main()
