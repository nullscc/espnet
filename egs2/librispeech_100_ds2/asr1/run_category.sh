#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev_clean"
# test_sets="test_clean test_other dev_clean dev_other"
test_sets="test_clean"

asr_tag=category
asr_config=conf/tuning/train_category.yaml
inference_config=conf/decode_asr.yaml

#--speed_perturb_factors "0.9 1.0 1.1" \
./category.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --lang en \
    --ngpu 1 \
    --nj 32 \
    --inference_nj 1 \
	--gpu_inference true \
    --token_type char \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --audio_format "wav" \
    --feats_type raw \
	--feats_normalize utt_mvn \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
	--inference_asr_model "valid.acc.best.pth" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
