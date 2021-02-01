#!/bin/bash
num_cores=10

echo "Running round 0"
python MCMC_w_dist_uncer.py /home/chengi/Documents/coop2021/data/hii_v2_20201203.db \
    --num_cores $num_cores --num_chains $num_cores --num_tune 2500 --num_iter 10000 \
    --num_samples 100 --prior_set A1 --like_type sivia --filter_plx False --num_round 0