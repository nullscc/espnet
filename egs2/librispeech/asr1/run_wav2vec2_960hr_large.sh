#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

# asr_config=conf/train_asr_conformer.yaml
asr_config=conf/tuning/train_asr_conformer7_wav2vec2_960hr_large.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

# --speed_perturb_factors "0.9 1.0 1.1" \
# --feats_normalize utt_mvn \
./asr.sh \
    --lang en \
    --ngpu 2 \
    --nbpe 5000 \
	--use_lm false \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --feats_normalize utt_mvn \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
	--inference_asr_model 15epoch.pth \
    --bpe_train_text "data/${train_set}/text" "$@"
