python -m DeBERTa.apps.run --model_config ./deberta-v3-large/rtd_large.json  \
	--tag deberta-v3-large \
	--do_train \
	--max_seq_len 320 \
	--task_name adv-qqp \
	--data_dir ./glue_data/QQP \
	--adv_data_path ./adv_glue/dev.json \
	--output_dir ./output/deberta-v3-large/QQP \
	--num_train_epochs 2 \
	--fp16 True \
	--warmup 1000 \
	--do_train \
	--learning_rate 1e-5  \
	--train_batch_size 8 \
	--cls_drop_out 0.2 \
	--cache_dir ./deberta-v3-large \
	--vocab_type spm \
	--vocab_path ./deberta-v3-large/spm.model \
	--vat_lambda 4 \
	--rtd_weight 1.5 \
	--init_generator ./deberta-v3-large/pytorch_model.generator.bin \
	--init_discriminator ./deberta-v3-large/pytorch_model.bin \