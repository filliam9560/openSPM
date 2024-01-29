#!/bin/ash
step=10 #interval cann't big than 60 
savedir=/opt/cont
cd "$savedir"
for ((i=0;i<=60;(i=i+step)));do 
	date    >>record_at
	free -k >>record_at
	echo "\n">>record_at
	sleep $step
done
exit 0
