#!/bin/bash
#source /local/mohammad/5HT2A/packages/nest-install/bin/nest_vars.sh
log_dir="LOG"
#rm -r $log_dir
mkdir $log_dir

for sim in {0..4}
do
    echo $sim
    rm "$log_dir/$sim.log" "$log_dir/$sim.err"
    python simulateNetworks.py $sim $1>"$log_dir/$sim.log" 2>"$log_dir/$sim.err" &
done
