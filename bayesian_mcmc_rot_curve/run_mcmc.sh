#!/bin/bash
num_cores=10

python MCMC_w_dist_uncer.py /home/chengi/Documents/coop2021/data/hii_v2_20201203.db \
    --num_cores $num_cores --num_chains $num_cores --num_tune 2500 --num_iter 5000 \
    --num_samples 100 --prior_set A1 --like_type cauchy \
    --num_rounds 1 --reject_method lnlike --filter_plx

# Other params:
# --filter_plx (will also filter sources with e_plx/plx > 0.2)
# --this_round (for setting starting point for filtering)