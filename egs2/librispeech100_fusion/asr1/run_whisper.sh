#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev_clean"
test_sets="test_clean_5db"

asr_tag=whisper
asr_config=conf/train_whisper.yaml
inference_config=conf/decode_asr.yaml

./asr_whisper.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --lang en \
    --ngpu 1 \
    --nj 8 \
    --inference_nj 1 \
    --nbpe 5000 \
 	--gpu_inference true \
	--expdir exp_whisper \
    --max_wav_duration 30 \
    --audio_format "wav" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
	--feats_normalize utt_mvn \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
