project=vietasr-ssl-alg10k-zipformer-2025-08-27
while true
do
  run=zipformer-medium_ce_4xH100_md4000_accgrad1_fp16_lr0.045_lrhours1e4_warmup1e3
  exp=zipformer_fbank/exp_iter2_h100
  wandb sync $exp/tensorboard/  -p $project  --id $run  --append
  sleep 600
done
