#!/bin/bash
#source /local/mohammad/5HT2A/packages/nest-install/bin/nest_vars.sh
log_dir="LOG"
#rm -r $log_dir
mkdir $log_dir

LO_BND=$(($1-1))
UP_BND=$(($2-1))
for (( sim=$LO_BND; sim<=$UP_BND; sim++))
do
    echo $sim
    rm "$log_dir/$sim.out" "$log_dir/$sim.err"
    python simulateNetworks.py $sim $3>"$log_dir/$sim.out" 2>"$log_dir/$sim.err" &
done
#python simulateNetworks.py $LO_BND $3>"$log_dir/$sim.out" 2>"$log_dir/$sim.err"
