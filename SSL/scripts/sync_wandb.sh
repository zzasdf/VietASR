while true
do
  project=vietasr-ssl-alg10k-zipformer-2025-08-27
  run=zipformer-medium_ce_4xH100_md4000_accgrad1_fp16_lr0.045_lrhours1e4_warmup1e3
  exp=zipformer_fbank/exp_iter2_h100
  wandb sync $exp/tensorboard/  -p $project  --id $run  --append

  project=vietasr-asr-alg10k-zipformer-2025-08-30
  run=epoch143avg123_zipformer-medium_prunedrnnt_4x4090_48G_md2400_accgrad1_fp16_lr2e-3
  exp=zipformer_fbank/exp_iter2_epoch143avg123_ft_ws4_md2400
  wandb sync $exp/tensorboard/  -p $project  --id $run  --append

  sleep 600
done
