python sample_generation.py \
--difficulty "advanced" \
--seed_setup 2 \
--max_iter 300 \
--no_padding \
--n_samples 10 \
--batch_size 10 \
--alpha 1000  \
--beta 1000 \
--gamma 100 \
--delta 500 \
--theta 500 \
--phi 0 \
--text_path_da "./inputs/a/text_a_d.txt" \
--topic_path_da "./inputs/a/topic_a_d.txt" \
--eq_path_da "./inputs/a/eq_a_d.txt" \
--model_path "./mlm_model/model_files_large_16_5_5e-05" \
--tok_path "./mlm_model/model_files_large_16_5_5e-05" \
--mlm_size "bert-large" \
--topic_dir "./topic_model/model_files_mod_32_5_2e-05" \



