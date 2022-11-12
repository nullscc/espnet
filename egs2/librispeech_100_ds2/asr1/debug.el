(dap-register-debug-template "run_ds2"
  (list :type "python"
        :args "--use_preprocessor true --bpemodel none --token_type char --token_list data/en_token_list/char/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none --valid_data_path_and_name_and_type dump/raw/dev_clean/wav.scp,speech,sound --valid_data_path_and_name_and_type dump/raw/dev_clean/text,text,text --valid_shape_file exp/asr_stats_raw_en_char/valid/speech_shape --valid_shape_file exp/asr_stats_raw_en_char/valid/text_shape.char --resume true --init_param --ignore_init_mismatch false --fold_length 80000 --fold_length 150 --output_dir exp/asr_ds2 --config conf/tuning/train_asr_ds2.yaml --frontend_conf fs=16k --train_data_path_and_name_and_type dump/raw/train_clean_100/wav.scp,speech,sound --train_data_path_and_name_and_type dump/raw/train_clean_100/text,text,text --train_shape_file exp/asr_stats_raw_en_char/train/speech_shape --train_shape_file exp/asr_stats_raw_en_char/train/text_shape.char --ngpu 1 --multiprocessing_distributed True --num_workers 0"
        :cwd "/ssdhome/xzw521/espnet-v.202205/egs2/librispeech_100_ds2/asr1"
        :env '(("CUDA_VISIBLE_DEVICES" . "9"))
        :target-module (expand-file-name "/ssdhome/xzw521/espnet-v.202205/espnet2/bin/asr_train.py")
        :request "launch"
        :name "run_ds2"))
