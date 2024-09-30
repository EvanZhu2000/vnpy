#!/bin/bash

# Assign the current date to date_str
date_str=$(date +%Y%m%d)

/home/evan/miniconda3/envs/vnpy3/bin/python /home/evan/miniconda3/envs/vnpy3/lib/python3.10/site-packages/vnpy_self/data_and_db/nth_rq_data_filler.py "dominant" $date_str $date_str
