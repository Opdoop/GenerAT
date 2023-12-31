python -m DeBERTa.apps.run --model_config ./deberta-v3-large/rtd_large.json  \
	--tag deberta-v3-large \
	--do_train \
	--task_name adv-rte \
	--data_dir ./glue_data/RTE \
	--adv_data_path ./adv_glue/dev.json \
	--output_dir ./output/deberta-v3-large/RTE \
	--num_train_epochs 3 \
	--fp16 True \
	--warmup 50 \
	--learning_rate 7e-6  \
	--train_batch_size 8 \
	--max_seq_len 320     \
	--cls_drop_out 0.3 \
	--cache_dir ./deberta-v3-large \
	--vocab_type spm \
	--vocab_path ./deberta-v3-large/spm.model \
	--vat_lambda 4 \
	--rtd_weight 1.5 \
	--init_generator ./deberta-v3-large/pytorch_model.generator.bin \
	--init_discriminator ./deberta-v3-large/pytorch_model.bin \
