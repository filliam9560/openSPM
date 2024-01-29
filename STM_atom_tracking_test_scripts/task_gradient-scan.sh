#!/bin/bash

############# 定义一个带有多个参数的atom_tracking函数
# 在需要时调用函数并传递多个参数:atom_tracking "n_loop" "n_point"
atom_tracking() {
  local n_loop=$1
  local n_point=$2
  echo "call atom_tracking function"
  ## add manipulation
  ## activate atom_tracking for some times
  open_at="0072"
  echo -e "$open_at\n.\n" | $w_control
  echo "\n"
  
  # atom_tracking time generate
  if (( $(echo "$radius < 10" | bc -l) )); then
      track_time=1
  elif (( $(echo "$radius > 10 " | bc -l) )); then
      track_time=$(echo "$radius / 10" | bc -l)
  fi
  milliseconds=$(echo "$track_time * 1000" | bc -l | cut -d '.' -f 1) #休眠毫秒数
  printf "wait %s milliseconds\n" $milliseconds
  sleep "$(echo "$milliseconds / 1000" | bc -l)" 
  #sleep $track_time #fell asleep for some seconds
  ## deactivate atom_tracking for some times
  shut_at="0070"
  echo -e "$shut_at\n.\n" | $w_control
  echo "\n"
  ## record x,y,z
  # 读取当前点
  xyz_rightnow=$(d 60000038 4 3) #read xyz right now
  x_central=$(echo -e ${xyz_rightnow:10:8} | sed -r 's/0*([0-9a-f])/\1/')
  y_central=$(echo -e ${xyz_rightnow:19:8} | sed -r 's/0*([0-9a-f])/\1/')
  z_central=$(echo -e ${xyz_rightnow:28:8} | sed -r 's/0*([0-9a-f])/\1/')
  printf "%8s %8s %8s \n" $x_central $y_central $z_central >> xyz_central
  # echo "中心点的坐标: ($x_central, $y_central, $z_central)"
  #
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

  echo "$n_loop $n_point 的坐标: ($x_int, $y_int)"
  x_output=$(printf "%x" $((16#$x_central + 10#$x_int))) 
  y_output=$(printf "%x" $((16#$y_central + 10#$y_int))) 
  # 写出
  echo -e "$x_output\n.\n" | $w_x 
  echo "\n"
  echo -e "$y_output\n.\n" | $w_y 
  echo "\n"

  sleep 1 #fell asleep for some seconds
  xyz_rightnow=$(d 60000038 4 3) #read xyz right now
  x_point=$(echo -e ${xyz_rightnow:10:8} | sed -r 's/0*([0-9a-f])/\1/')
  y_point=$(echo -e ${xyz_rightnow:19:8} | sed -r 's/0*([0-9a-f])/\1/')
  z_point=$(echo -e ${xyz_rightnow:28:8} | sed -r 's/0*([0-9a-f])/\1/')
  printf "%8s %8s %8s \n" $x_point $y_point $z_point >> xyz_point
  # echo "扫描点的坐标: ($x_point, $y_point, $z_point)"

  z_delta=$(printf "%x" $((16#$x_central - 16#$x_point))) 
  printf "%8s %8s %8s \n" $x_int $y_int $z_delta >> xyz_relative
}

#######according to the mode to determine base name
save_fig() {
  # echo "call save_fig function"
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

  
  file_xyz_central="${base_file_name}_central_scan${n_pic}.txt"
  file_xyz_point="${base_file_name}_point_scan${n_pic}.txt"
  file_xyz_relative="${base_file_name}_relative_scan${n_pic}.txt"
  touch $file_xyz_central
  touch $file_xyz_point
  touch $file_xyz_relative
  echo "$content_head" >> $file_xyz_central
  echo "$content_head" >> $file_xyz_point
  echo "$content_head" >> $file_xyz_relative
  cat xyz_central >> $file_xyz_central
  cat xyz_point >> $file_xyz_point
  cat xyz_relative >> $file_xyz_relative
  (d 60000000 2 60) >> $file_xyz_central
  (d 60000000 2 60) >> $file_xyz_point
  (d 60000000 2 60) >> $file_xyz_relative

}



########### __main__###############
savedir=/opt/cont
cd "$savedir" || exit 1 # Exit script if cd fails
content_head=$(printf "%8s %8s %8s \n" 'x' 'y' 'z' )

w_control="m 6000000c 2"
w_x="m 60000038 4 1"
w_y="m 6000003c 4 1"

n_pic=0
n_pic_max=2        #作图数
n_loop=0
n_loop_max=5      # 圈数
n_point=0
n_point_max=5  # 一圈的离散点数量
radius_max=50  # 最大半径限制 100


# 外部循环
for ((n_pic = 0; n_pic < n_pic_max; n_pic++)); do
  rm xyz_central
  rm xyz_point
  rm xyz_relative
  # 内部循环
  for ((n_loop = 0; n_loop < n_loop_max; n_loop++)); do
    for ((n_point = 0; n_point < n_point_max; n_point++)); do
      # 计算总圈数对应的角度增量和半径增量
      n_point_total=$(echo "$n_point_max * $n_loop_max" | bc -l)
      # 逆时针旋转，角度增量为正数
      angle_increment=$(echo "scale=4; 360 / $n_point_max" | bc -l) 
      radius_increment=$(echo "scale=4; $radius_max / $n_point_total" | bc -l)
      # 计算半径和相位
      radius=$(echo "$radius_increment * ( ($n_loop * $n_point_max) + ($n_point) )" | bc -l) # 逆时针旋转，角度转换为弧度，并乘以-1
      theta=$(echo "$angle_increment * $n_point" | bc -l)
      atom_tracking "$n_loop" "$n_point"
      move
    done
  done
  save_fig
done

