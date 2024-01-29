#!/bin/bash

savedir=/opt/cont
cd "$savedir"
content_head="addr      x      y      null      null"

#according to the mode to determine base name
mode=$(d 600000a0 2 1)#
if [ ${mode:12:1}==0 ]
then
mode_scan=radius
fi
if [ ${mode:12:1}==1 ]                                             
then                                                               
mode_scan=helix                                                   
fi
if [ ${mode:13:1}==0 ]                                             
then                                                               
mode_algorithm=extremum                                                   
fi                                                                 
if [ ${mode:12:1}==4 ]                                             
then                                                               
mode_algorithm=gradient                                                    
fi  
base_file_name="record_at_$mode_algorithm_$mode_scan_drift"


name_step=$(d 6000008c 2 1)#name=4
name_r=$(d 60000088 2 1)#name=4
num_step=$(echo -e ${name_step:10:4} | sed -r 's/0*([0-9a-f])/\1/')
num_r=$(echo -e ${name_r:10:4} | sed -r 's/0*([0-9a-f])/\1/')
delta_step=8
delta_r=120
threshold=120
file_xy="${base_file_name}${name_r:10:4}_${name_step:10:4}.txt"
file_z="${base_file_name}${name_r:10:4}_${name_step:10:4}z.txt"
touch $file_xy
touch $file_z
echo $content_head >> $file_xy
echo $content_head >> $file_z
cat record_xy >> $file_xy
cat record_z  >> $file_z
(d 60000000 2 60) >> $file_xy
(d 60000000 2 60) >> $file_z

#new_step=`expr $num_step + $delta_step`                        
#new_r=`expr $num_r + $delta_r`                             
new_step=`printf "%x" $((16#$num_step + 16#$delta_step))`
new_r=`printf "%x" $((16#$num_r + 16#$delta_r))`                                                     

if [ $((16#$new_step)) -le $((16#$threshold)) ]                                        
then                                                 
change_step="m 6000008c 2"                           
echo -e "$new_step\n.\n" | $change_step                    
rm record_xy                                          
rm record_z                                           
fi                                                    
                                                      
if [ $((16#$new_step)) -gt $((16#$threshold)) ]                                         
then                                                  
change_r="m 60000088 2"                      
change_step="m 6000008c 2"                   
echo -e "$new_r\n.\n" | $change_r  
echo -e "0004\n.\n" | $change_step
rm record_xy                               
rm record_z                                          
fi 

#find -type f -name "re*.txt" -exec rm {} \;

