#!/bin/bash

savedir=/opt/cont
cd "$savedir"
content_head="addr      x      y      null      null"
base_file_name=record_at_extremum_radius_drift0368
name=$(d 6000008c 2 1)#name=4
num=$(echo -e ${name:10:4} | sed -r 's/0*([0-9])/\1/')
delta=4
file_xy="${base_file_name}_${name:10:4}.txt"
file_z="${base_file_name}_${name:10:4}z.txt"
touch $file_xy
touch $file_z
echo $content_head >> $file_xy
echo $content_head >> $file_z
cat record_xy >> $file_xy
cat record_z  >> $file_z
(d 60000000 2 60) >> $file_xy
(d 60000000 2 60) >> $file_z
new=`expr $num + $delta` 
change="m 6000008c 2" 
echo -e "$new\n.\n" | $change
rm record_xy
rm record_z
#find -type f -name "re*.txt" -exec rm {} \;

