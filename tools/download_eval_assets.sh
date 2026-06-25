#!/usr/bin/env bash
# Download the offline evaluation assets without requiring Python or Conda.

set -euo pipefail

: "${EVAL_ASSET_ROOT:?Set EVAL_ASSET_ROOT to an external asset directory}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MS_ENDPOINT="${MS_ENDPOINT:-https://modelscope.cn/models}"
ARIA2C="${ARIA2C:-aria2c}"

UTMOS_REV="506474f2b33dc77c234d668cc419be1861899cad"
WAV2VEC_REV="0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
WESPEAKER_REV="ed25c1ecbd9e8d3c7944df95430da02bd3cdcf7a"

download() {
    local url="$1"
    local destination="$2"
    mkdir -p "$(dirname "$destination")"
    "$ARIA2C" -c -x 8 -s 8 -d "$(dirname "$destination")" \
        -o "$(basename "$destination")" "$url"
}

UTMOS_ROOT="$EVAL_ASSET_ROOT/utmosv2/models/fusion_stage3"
HF_SNAPSHOT="$EVAL_ASSET_ROOT/huggingface/hub/models--facebook--wav2vec2-base/snapshots/$WAV2VEC_REV"
FUNASR_ROOT="$EVAL_ASSET_ROOT/funasr/Fun-ASR-Nano-2512"
VAD_ROOT="$EVAL_ASSET_ROOT/funasr/fsmn-vad"
WESPEAKER_ROOT="$EVAL_ASSET_ROOT/wespeaker/cnceleb-resnet34-LM"

download "$HF_ENDPOINT/sarulab-speech/UTMOSv2/resolve/$UTMOS_REV/fold0_s42_best_model.pth" \
    "$UTMOS_ROOT/fold0_s42_best_model.pth"

for name in config.json preprocessor_config.json pytorch_model.bin \
    special_tokens_map.json tokenizer_config.json vocab.json; do
    download "$HF_ENDPOINT/facebook/wav2vec2-base/resolve/$WAV2VEC_REV/$name" \
        "$HF_SNAPSHOT/$name"
done
mkdir -p "$EVAL_ASSET_ROOT/huggingface/hub/models--facebook--wav2vec2-base/refs"
printf '%s\n' "$WAV2VEC_REV" \
    > "$EVAL_ASSET_ROOT/huggingface/hub/models--facebook--wav2vec2-base/refs/main"

download "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx" \
    "$EVAL_ASSET_ROOT/dnsmos/DNSMOS/model_v8.onnx"
download "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx" \
    "$EVAL_ASSET_ROOT/dnsmos/DNSMOS/sig_bak_ovr.onnx"
download "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/pDNSMOS/sig_bak_ovr.onnx" \
    "$EVAL_ASSET_ROOT/dnsmos/pDNSMOS/sig_bak_ovr.onnx"

for name in config.yaml configuration.json model.pt multilingual.tiktoken; do
    download "$MS_ENDPOINT/FunAudioLLM/Fun-ASR-Nano-2512/resolve/master/$name" \
        "$FUNASR_ROOT/$name"
done
for name in config.json generation_config.json merges.txt tokenizer.json \
    tokenizer_config.json vocab.json; do
    download "$MS_ENDPOINT/FunAudioLLM/Fun-ASR-Nano-2512/resolve/master/Qwen3-0.6B/$name" \
        "$FUNASR_ROOT/Qwen3-0.6B/$name"
done

for name in am.mvn config.yaml configuration.json model.pt; do
    download "$MS_ENDPOINT/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/resolve/master/$name" \
        "$VAD_ROOT/$name"
done

for name in config.yaml model_5.pt; do
    download "$HF_ENDPOINT/Wespeaker/wespeaker-cnceleb-resnet34-LM/resolve/$WESPEAKER_REV/$name" \
        "$WESPEAKER_ROOT/$name"
done

echo "Evaluation assets downloaded under: $EVAL_ASSET_ROOT"
echo "Validate after the Python environment exists with:"
echo "  python tools/check_track2_setup.py --eval-asset-root \"$EVAL_ASSET_ROOT\""
