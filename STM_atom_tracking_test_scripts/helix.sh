#!/bin/bash

# 控制参数
radius_max=100       # 最大半径限制
n_point_max=100             # 一圈的离散点数量
n_loop_max=5      # 圈数

# 计算总圈数对应的角度增量和半径增量
n_point_total=$(echo "$n_point_max * $n_loop_max" | bc -l)
angle_increment=$(echo "scale=4; 360 / $n_point_max" | bc -l) # 逆时针旋转，角度增量为正数
radius_increment=$(echo "scale=4; $radius_max / $n_point_total" | bc -l)
# 循环生成离散点
for ((n_loop = 0; n_loop < n_loop_max; n_loop++)); do
  for ((n_point = 0; n_point < n_point_max; n_point++)); do
      theta=$(echo "$angle_increment * $n_point" | bc -l)

      # 计算半径
      radius=$(echo "$radius_increment * ( ($n_loop * $n_point_max) + ($n_point) )" | bc -l) # 逆时针旋转，角度转换为弧度，并乘以-1

      # 如果半径超过最大半径限制，则停止增加半径
      if (( $(echo "$radius < 0" | bc -l) )); then
          radius=0
      elif (( $(echo "$radius > $radius_max" | bc -l) )); then
          radius=$radius_max
      fi

      # 计算点的坐标
      x=$(echo "$radius * c($theta * 0.0174533)" | bc -l)
      y=$(echo "$radius * s($theta * 0.0174533)" | bc -l)

      # 将浮点数坐标转换为整数
      x_int=$(printf "%.0f" "$x")  # 四舍五入转换为整数
      y_int=$(printf "%.0f" "$y")  # 四舍五入转换为整数

      echo "$n_loop $n_point 的坐标: ($x_int, $y_int)"
  done
done