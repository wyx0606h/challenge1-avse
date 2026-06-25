# Offline evaluation asset manifest

This document records the external model assets prepared for Track 2 baseline
reproduction. The files themselves live outside Git under
`EVAL_ASSET_ROOT`.

Preparation date: 2026-06-26

## Layout

```text
<EVAL_ASSET_ROOT>/
├── utmosv2/
├── huggingface/
├── dnsmos/
├── funasr/
│   ├── Fun-ASR-Nano-2512/
│   └── fsmn-vad/
└── wespeaker/cnceleb-resnet34-LM/
```

Total prepared size: approximately 3.2 GiB.

## Source revisions and primary checksums

| Asset | Source revision | Primary file | Size | SHA256 |
|---|---|---|---:|---|
| UTMOSv2 | `506474f2b33dc77c234d668cc419be1861899cad` | `fold0_s42_best_model.pth` | 818,531,314 | `c8149d988e4bbf3f347e6966b5d769de347a5f8c59ffca1dc4bd4bf5b8585e57` |
| facebook/wav2vec2-base | `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8` | `pytorch_model.bin` | 380,267,417 | `3249fe98bfc62fcbc26067f724716a6ec49d12c4728a2af1df659013905dff21` |
| DNSMOS | Microsoft DNS-Challenge `master` at preparation time | `DNSMOS/model_v8.onnx` | 224,860 | `9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91` |
| DNSMOS | Microsoft DNS-Challenge `master` at preparation time | `DNSMOS/sig_bak_ovr.onnx` | 1,157,965 | `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd` |
| DNSMOS personalized | Microsoft DNS-Challenge `master` at preparation time | `pDNSMOS/sig_bak_ovr.onnx` | 1,157,962 | `9e3a197449ca2177f0997afec3bd6b890117ce2f17b89d6eea7fa0d47272c81c` |
| Fun-ASR-Nano-2512 | ModelScope `master` at preparation time | `model.pt` | 2,127,426,538 | `81fec8616083c69377f3ceef36aba3655660ee0ca69a5d4a1e9810cd340ca499` |
| fsmn-vad | ModelScope `master` at preparation time | `model.pt` | 1,721,366 | `b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5` |
| WeSpeaker cnceleb-resnet34-LM | `ed25c1ecbd9e8d3c7944df95430da02bd3cdcf7a` | `model_5.pt` | 35,222,731 | `4fed1fbf1bb1e772f794e54f7dd8c5635485a0ccce799ecb429a9232d22596f7` |

The Fun-ASR directory also contains its configuration,
`multilingual.tiktoken`, and the complete `Qwen3-0.6B` configuration/tokenizer
files. The wav2vec2 snapshot includes its model and processor configuration.

## Validation

The file-level preflight completed with 19 checks passing:

```bash
python tools/check_track2_setup.py \
  --eval-asset-root "$EVAL_ASSET_ROOT"
```

No `.aria2` partial-download files remained after preparation.

This is a file-integrity check, not a runtime model-load claim. Runtime loading
must be verified after the isolated Python/CUDA environment is installed.

## Re-download

The repository provides a Python-independent downloader:

```bash
export EVAL_ASSET_ROOT=/external/path/to/avse-assets/evaluation
HF_ENDPOINT=https://hf-mirror.com bash tools/download_eval_assets.sh
```

Never commit these assets or bypass `.gitignore` protections.
