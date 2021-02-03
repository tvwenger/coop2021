python MCMC_w_dist_uncer.py /home/chengi/Documents/coop2021/data/hii_v2_20201203.db \
    --num_cores 2 --num_chains 2 --num_tune 500 --num_iter 500 \
    --num_samples 100 --prior_set A5 --like_type cauchy \
    --num_rounds 2 --reject_method lnlike --filter_plx

# Other params:
# --filter_plx (will also filter sources with e_plx/plx > 0.2)
# --this_round (for setting starting point for filtering)