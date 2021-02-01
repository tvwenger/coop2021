python MCMC_w_dist_uncer.py /home/chengi/Documents/coop2021/data/hii_v2_20201203.db \
    --num_cores 2 --num_chains 2 --num_tune 100 --num_iter 100 \
    --num_samples 5 --prior_set A1 --like_type sivia --filter_plx False \
    --num_rounds 2 --filter_method sigma