#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev_clean"
test_sets="test_clean dev_clean"

asr_tag=nat_thin
asr_config=conf/tuning/nat_thin.yaml
inference_config=conf/decode_asr.yaml
train_npy_scp=npy_thin.scp

# --inference_asr_model "valid.acc.best.pth" \
./nat.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --lang en \
    --ngpu 2 \
    --nj 8 \
    --inference_nj 1 \
	--gpu_inference true \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --audio_format "wav" \
	--feats_normalize utt_mvn \
    --feats_type raw \
	--train_npy_scp ${train_npy_scp} \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
