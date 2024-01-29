#!/bin/bash

############# 定义一个带有多个参数的atom_tracking函数
# 在需要时调用函数并传递多个参数:atom_tracking "n_radius" "n_step_length"
atom_tracking_open() {
  echo "open atom_tracking function"
  ## add manipulation
  open_at="0072"
  echo -e "$open_at\n.\n" | $w_control
  echo "\n"
}

atom_tracking_shut() {
  echo "shut atom_tracking function"
  ## add manipulation
  shut_at="0070"
  echo -e "$shut_at\n.\n" | $w_control
  echo "\n"
}

hex=c
for ((i = 1; i <= $(echo "ibase=16; ${hex^^}" | bc); i++)); do
    decimal=$(printf "%d" "$i")  # 将十六进制转换为十进制
    echo $decimal
done


record() {
  echo "call record function"
  ## add manipulation
  interval=10 #单位微秒
  n_record_max=100 #记录次数
  for ((n_record = 0; n_record < n_record_max; n_record++)); do
    ## record x,y,z
    # 读取当前点
    xyz_rightnow=$(d 60000038 4 3) #read xyz right now
    x_central=$(echo -e ${xyz_rightnow:10:8} | sed -r 's/0*([0-9a-f])/\1/')
    y_central=$(echo -e ${xyz_rightnow:19:8} | sed -r 's/0*([0-9a-f])/\1/')
    z_central=$(echo -e ${xyz_rightnow:28:8} | sed -r 's/0*([0-9a-f])/\1/')
    printf "%8s %8s %8s \n" $x_central $y_central $z_central >> rate_test
    # echo "中心点的坐标: ($x_central, $y_central, $z_central)"
    usleep $interval
  done
}

change_mode() {
  echo "call change_mode function"
  # add manipulation

  # 计算大小(多乘以4，是为了可以直接写入DA当中)
  n_radius4da=$(echo "$n_radius * 4" | bc -l)
  n_step_length4da=$(echo "$n_step_length * 4" | bc -l)

  echo -e "$n_radius4da\n.\n" | $w_radius 
  echo "\n"
  echo -e "$n_step_length4da\n.\n" | $w_step_length 
  echo "\n"
}

############# 定义一个带有多个参数的move函数
move() {
  echo "call move function"
  # add manipulation

  # 计算点的坐标(多乘以4，是为了可以直接写入DA当中)
  x=$(echo "$radius * c($theta * 0.0174533) * 4" | bc -l)
  y=$(echo "$radius * s($theta * 0.0174533) * 4" | bc -l)

  # 将浮点数坐标转换为整数
  x_int=$(printf "%.0f" "$x")  # 四舍五入转换为整数
  y_int=$(printf "%.0f" "$y")  # 四舍五入转换为整数

  echo "$n_radius $n_step_length 的相对坐标: ($x_int, $y_int)"
  x_output=$(printf "%x" $((16#$x_central + 10#$x_int))) 
  y_output=$(printf "%x" $((16#$y_central + 10#$y_int))) 
  echo "位移到的坐标: ($x_output, $y_output)"
  # 写出
  echo -e "$x_output\n.\n" | $w_x 
  echo "\n"
  echo -e "$y_output\n.\n" | $w_y 
  echo "\n"

  sleep 1 #fell asleep for some seconds



}

#######according to the mode to determine base name
save_data() {
  echo "call save_data function"
  # add manipulation
  mode=$(d 600000a0 2 1)#
  if [ ${mode:12:1} == 0 ]
  then
  mode_scan=radius
  fi
  if [ ${mode:12:1} == 1 ]                                             
  then                                                               
  mode_scan=helix                                                   
  fi
  if [ ${mode:13:1} == 0 ]                                             
  then                                                               
  mode_algorithm=extremum                                                   
  fi                                                                 
  if [ ${mode:13:1} == 4 ]                                             
  then                                                               
  mode_algorithm=gradient                                                    
  fi  
  name_step_length=$(d 6000008c 2 1) #name=4
  name_r=$(d 60000088 2 1) #name=4
  #num_step_length=$(echo -e ${name_step_length:10:4} | sed -r 's/0*([0-9a-f])/\1/')
  #num_r=$(echo -e ${name_r:10:4} | sed -r 's/0*([0-9a-f])/\1/')
  base_file_name="${mode_algorithm}_${mode_scan}_${name_r:10:4}_${name_step_length:10:4}"

  
  file_rate_test="${base_file_name}_rate_test${n_pic}.txt"
  touch $file_rate_test
  echo "$content_head" >> $file_rate_test
  cat rate_test >> $file_rate_test
  (d 60000000 2 60) >> $file_rate_test
}



########### __main__###############
savedir=/opt/cont
cd "$savedir" || exit 1 # Exit script if cd fails
content_head=$(printf "%8s %8s %8s \n" 'x' 'y' 'z' )

w_control="m 6000000c 2"
w_x="m 60000038 4 1"
w_y="m 6000003c 4 1"
w_radius="m 60000088 2"
w_step_length="m 6000008c 2"

n_pic=0
n_pic_max=2        #
n_radius=0
n_radius_max=20      # 
n_step_length=0
n_step_length_max=5  # 
radius=100 #单位LSB
theta=45 # 单位度数

# 外部循环
for ((n_pic = 0; n_pic < n_pic_max; n_pic++)); do

  # 内部循环
  for ((n_radius = 5; n_radius < n_radius_max; n_radius++)); do
    for ((n_step_length = 1; n_step_length < n_step_length_max; n_step_length++)); do
      rm rate_test
      atom_tracking_shut
      change_mode
      move
      atom_tracking_open
      record
      sleep 10
      save_data
    done
  done
done

