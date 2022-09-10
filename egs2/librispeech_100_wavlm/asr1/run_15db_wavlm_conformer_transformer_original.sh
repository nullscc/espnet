#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train100"
valid_set="dev"
test_sets="test_clean_original test_15db test_0db"

asr_tag=15db_wavlm_conformer_transformer_original
asr_config=conf/tuning/15db_wavlm_conformer_transformer_original.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --lang en \
    --ngpu 2 \
    --nj 32 \
    --inference_nj 32 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --audio_format "wav" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
	--inference_asr_model 19epoch.pth \
	--feats_normalize utt_mvn \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
