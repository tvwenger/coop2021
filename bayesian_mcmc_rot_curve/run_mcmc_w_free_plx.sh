#!/bin/bash
num_cores=10

echo "==== MCMC w/ plx as model parameter (truncated normal) + mean Upec & Vpec ===="
echo "Cauchy PDF then Gaussian PDF for round 2+"

echo "=== Testing real data ==="
python mcmc_w_free_plx.py /home/chengi/Documents/coop2021/data/hii_v2_20201203.db \
    --num_cores $num_cores --num_chains $num_cores --num_tune 2500 --num_iter 12000 \
    --prior_set A1 --like_type cauchy --num_rounds 3 --reject_method lnlike \
    --free_Zsun --free_roll --auto_run


# echo "==== Testing simulated data ===="
# python MCMC_w_dist_uncer.py \
#     /home/chengi/Documents/coop2021/bayesian_mcmc_rot_curve/mcmc_sim_data.pkl \
#     --num_cores $num_cores --num_chains $num_cores --num_tune 1000 --num_iter 1000 \
#     --num_samples 100 --prior_set A5 --like_type cauchy \
#     --num_rounds 3 --reject_method lnlike --free_Zsun --free_roll --auto_run

# Other params:
# --filter_plx (will also filter sources with e_plx/plx > 0.2)
# --this_round (for setting starting point for filtering)
# --free_Wpec (let Wpec be free parameter)
# --free Zsun (let Zsun be free parameter)
# --free_roll (let roll angle b/w galactic midplane &
#              galactocentric frame be free parameter)
# --auto_run (let MCMC program run until no more outliers rejected.
#             Will override num_rounds)
# --individual_Upec (let Upec vary per source)
# --individual_Vpec (let Vpec vary per source)