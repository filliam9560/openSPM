#!/bin/bash
step=2 #interval cann't big than 60
savedir=/opt/cont
cd "$savedir"
for ((i=0;i<=60;(i=i+step)));do

	d 60000098 4 2 >>record_xy
	d 60000040 4 1 >>record_z

        sleep $step
done
exit 0

