import matplotlib as mpl
import matplotlib.pyplot as plt
from colorspacious import cspace_converter
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import myBin2dec2hex as myfunc
import matplotlib.animation as animation
import os   #导入os模块
import glob
from matplotlib.animation import FFMpegWriter
import traceback
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
import copy
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit



animation_flag  =0
start_init      =0
end_init        =0  #所想绘制的数据点的绝对index
show            =1
save_pic_type=0 # 0:png 1:svg
point_minute = 30 # 一分钟记录30个点 一分钟记录一次
#################待处理的文件夹#########################
mode_algorithm =0 # 0:'extremum' 1:'gradient'
mode_scan      =1 # 0:'radius'   1:'helix'
mode_measure   =0
# date           ='1019z'

file_choose=1 # 1):name_all#遍历全部文件 2):name_list 3):name_files = glob.glob(re_file)
name_list=['record_at_gradient_helix_drift0024_0010.csv','record_at_gradient_helix_drift00a4_0008.csv']
re_file='*rec*.csv'#相应文件的正则表达式


algorithm='';scan='';measure='';
list_dir_sub_dic={
'0,0':          [
              's0'
              ,'s1'
              ,'s2'
              ,'s3'
              ]  ,
"0,1":         [
              's0'
              ,'s1'
              ,'s2'
              ,'s3'
              ]  ,
"1,0":           ['64data'
              ,'1024data'
              ,'1024data1111'
              ,'1024data1209'
              ,'1024data1211'
              ,'1024data1212'
              ,'1024data1213'
              ,'1024data1214'
              ,'s0'
              ,'s1'
              ,'s2'
              ,'s3'
              ]     ,
"1,1":           ['1024data1112'
              ,'1024data1121'
              ,'1024data1125'
              ,'1024data1127'
              ,'s0'
              ,'s1'
              , 's2'
              , 's3'
              ,'s4'
              ]
}
list_dir_sub=list_dir_sub_dic[str(mode_algorithm)+','+str(mode_scan)]
if mode_algorithm==0:
    algorithm='extremum'
else:
    algorithm='gradient'
if mode_scan==0:
    scan='radius'
else:
    scan='helix'
if mode_measure==0:
    measure='drift'
else:
    measure='slope'
name_dir = 'vary_'+algorithm+'_'+scan # 'record_at_'+algorithm+'_'+mode+date

######################### 颜色映射关系#########################################
####rgb
# startcolor  = (1.0, 0.0, 0.0)  #
# midcolor    = (0.0, 1.0, 0.0)  #
# endcolor    = (0.0, 0.0, 1.0)  #
####蓝色调
# startcolor  = (0.0, 0.0, 0.5)  #
# midcolor    = (0.0, 0.5, 0.5)  #
# endcolor    = (0.5, 0.5, 0.5)  #
#####灰度
startcolor = (0.1, 0.1, 0.1)  #
midcolor = (0.5, 0.5, 0.5)  #
endcolor = (0.85, 0.85, 0.85)  #
my_cmap1 = mpl.colors.LinearSegmentedColormap.from_list(name='mycmp1', colors=[startcolor, midcolor, endcolor])
mpl.colormaps.register(name='mycmp1', cmap=my_cmap1)

#############################################################################
for index,dir_sub in enumerate(list_dir_sub):
    path = 'C:/Users/14170/Desktop/python/atom_tracking_data_analysis/data/'+name_dir+'/'+dir_sub+'/'
    print('需要处理的文件所在的路径 %s' % path)
    path_pic='C:/Users/14170/Desktop/python/atom_tracking_data_analysis/pic/'+name_dir+'/'+dir_sub+'/'
    if not os.path.exists(path_pic):
        os.makedirs(path_pic)
    print('图片存储路径为 %s' % path_pic)
    # path_output = 'C:/Users/14170/Desktop/python/atom_tracking_data_analysis/pic' #需要处理的文件所在的路径
    os.chdir(path)  # 修改当前工作目录
    retval = os.getcwd()  # 查看当前工作目录
    print('当前工作目录为 %s' % retval)
    name_all = os.listdir()        #读取文件初始的名字 path为空则代表当前路径下
    print(name_all)
    name_files=''
    if(file_choose==1):
        name_files = name_all#遍历全部文件
    elif(file_choose==2):
        name_files = name_list
    elif(file_choose==3):
        name_files = glob.glob(re_file)

    flag_error='全部处理成功'
    name_wrong=''

    ###############################################################
    for name_full in name_files:
        try:
            print('待处理的数据' + name_full)
            portion = os.path.splitext(name_full)
            name = portion[0]
            # 如果后缀是.txt
            if ((portion[1] == ".csv")&(portion[0][-1]!='z')):
                print('正在处理 ',name_full)
                # input
                data_name_and_path = name + '.csv'
                # output
                pic_name_and_path = path_pic + name + '.png'#'.svg'
                gif_name_and_path = path_pic + name + '.gif'


                # def function(x):  # input is 22bits orignal DAC data,2 LSB is oversample
                #     x_bin = myfunc.hex2bin(x, 26)
                #     right_shift_num = 2  # 符号位要在n byte的最左边
                #     x_shift = x_bin + right_shift_num * '0'
                #     x_sign = myfunc.signed_bin2dec(x_shift)
                #     voltage = (x_sign / (4 * (2 ** right_shift_num))) * (20 / (2 ** 20))
                #     movment = voltage * 20  # nm/V
                #     return movment  # DAC output transform to movement of tip
                def function(x):  # input is orignal DAC data string,2 LSB is oversample,4 MSB is control bits
                    x_bin = myfunc.hex2bin(x, 26)   # x='3ffffff' x_bin='0b1111 11111111111111111111  11'
                    x_bin='0b'+x_bin[6:26]          #             x_bin='0b11111111111111111111'
                    right_shift_num = 0             # 符号位要在n byte的最左边
                    hva=4 # 高压放大倍数
                    DAC_bits=20 #DAC输出位宽
                    DAC_output_scale = 20 # DAC输出电压幅值为正负10V
                    x_shift = x_bin + right_shift_num * '0'
                    x_sign = myfunc.signed_bin2dec(x_shift)
                    voltage = (x_sign /  (2 ** right_shift_num)) * (DAC_output_scale / (2 ** DAC_bits)) * hva
                    movment = voltage * 20  # nm/V
                    return movment  # DAC output transform to movement of tip

                dac_20bits = '00011'  # '6d3'
                dac_20bits_bin = myfunc.hex2bin(dac_20bits, 20)
                dac_22bits_bin = dac_20bits_bin + 2 * '0'
                dac_22bits = myfunc.bin2hex(dac_22bits_bin, 6)
                print("dac_22bits is:", dac_22bits, "\ndac_movement_value is: ", function(dac_22bits), "nm")

                # 读文件
                data = pd.read_csv(data_name_and_path)
                # print(data.dtypes)
                data['x'] = data['x'].astype(str)
                data['y'] = data['y'].astype(str)
                # print(data.dtypes)
                # data['x'] = data['x'].apply(int, base=16).map(function)
                # data['y'] = data['y'].apply(int, base=16).map(function)
                data['x'] = data['x'].map(function)
                data['y'] = data['y'].map(function)
                # print(data.dtypes)
                # print(data.info)
                start=0;end=0
                if (start_init!=0):
                    start=start_init
                else:start=0
                if (end_init!=0):
                    end=end_init
                else:end=(data.shape[0]-1)
                # print('end=',end ,'data.shape=',data.shape)
                xdata = data.loc[start:end, 'x']  # 横坐标 时间是列名
                ydata = data.loc[start:end, 'y']

                xdata_max=max(xdata);xdata_min=min(xdata)
                ydata_max=max(ydata);ydata_min=min(ydata)
                colors = np.arange(0,1,(1-0)/(end-start+1))#从0开始太淡了
                # 如果想得到小数数列的话可以用numpy中的arange函数，
                # 自带的range函数只能得到整数类型的序列（注意当需要小数序列时用该函数会报错）。
                rgb = mpl.colormaps['Blues'](colors)[np.newaxis, :, :3]
                # print('colors= ',colors,'rgb= ',rgb,'type(rgb)= ',type(rgb))
                lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
                # print('lab= ',lab, 'type(lab)= ',type(lab))



                ###############################################################################
                #画图配置
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示乱码问题
                plt.rcParams['font.family'] = 'sans-serif'  # 设置字体格式
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.size'] = 12
                # 初始化画布
                fig = plt.figure(figsize=(11,8))
                ax=fig.add_subplot(111)


                ###########################################################################################
                # define func: y=ax+b
                def func(x, a, b):
                    x = np.array(x)
                    return a * x + b


                popt, pcov = curve_fit(func, xdata, ydata)
                print('popt=', popt, 'pcov=', pcov, '\n')
                y_fit = func(xdata, popt[0], popt[1])
                plt.plot(xdata, y_fit, 'k')

                mean = np.mean(ydata)  # 1.y mean
                ss_tot = np.sum((ydata - mean) ** 2)  # 2.total sum of squares
                ss_res = np.sum((ydata - func(xdata, *popt)) ** 2)  # 3.residual sum of squares
                r_squared = 1 - (ss_res / ss_tot)  # 4.r squared
                print('mean=', mean, 'ss_tot', ss_tot, 'ss_res', ss_res, 'r_squared', r_squared)
                ########################################################################################################

                # For a sequence of values to be color-mapped, use the 'c' argument instead.
                # scatter支持而plt.plot并不支持
                # ax0=ax[0].scatter只有一个ax的时候，不支持索引
                colors = np.arange(len(xdata))  # np.squeeze(xdata)
                ax0=ax.scatter(xdata, ydata, c=colors, cmap=mpl.colormaps.get_cmap('mycmp1'),alpha = 8/10,s=66
                                   , label = 'x_fit %.3f pm/min \ny_fit :%.3f pm/min \nx_max :%.3f pm/min\ny_max :%.3f pm/min \nr_squared :%.2f' % (
                    1000*(xdata[start] - xdata[end]) / ((end - start) / point_minute),
                    1000*(y_fit[start] - y_fit[end]) / ((end - start) / point_minute),
                    1000*(xdata_max - xdata_min) / ((end - start) / point_minute),
                    1000*(ydata_max - ydata_min) / ((end - start) / point_minute),
                    r_squared
                ))
                # 其中参数loc用于设置legend的位置，bbox_to_anchor用于在bbox_transform坐标（默认轴坐标）中为图例指定任意位置。
                plt.legend(loc=0, bbox_to_anchor=(0.37, 0.8), borderaxespad=0)#(0.37, 0.8)  (0.4, 0.3)


                norm = mcolors.Normalize(vmin=0, vmax=((end - start) / point_minute))
                im = cm.ScalarMappable(norm=norm, cmap=mpl.colormaps.get_cmap('mycmp1'))
                cb = fig.colorbar(im, ax=ax, orientation='vertical', ticks=np.linspace(0, ((end - start) / point_minute), 11),
                                  aspect=30)  # plt.cm.ScalarMappable(cmap=mpl.colormaps.get_cmap('mycmp1'))
                cb.set_label('Time(min)', fontsize=26)  # $\mathrm{(^oC)}$
                cb.outline.set_linewidth(0.5)

                plt.title(name+"\'thermal drift", fontsize=16)  # 曲线标题
                plt.xlabel(u'X/nm', fontsize=26)
                # 设置x轴标签旋转角度和字体大小
                plt.xticks(rotation=10, fontsize=16)
                plt.ylabel(u'Y/nm', fontsize=26)
                plt.yticks(rotation=0, fontsize=16)
                ticker_x=abs((xdata_max - xdata_min)/5)
                ticker_y=abs((ydata_max - ydata_min)/5)
                plt.gca().xaxis.set_major_locator(ticker.MultipleLocator( ticker_x ))  # 设置横坐标间隔（每隔ticker_x个横坐标显示一个横坐标，解决横坐标显示重叠问题）
                if save_pic_type==0:
                    plt.savefig(pic_name_and_path + '.png')
                else:
                    plt.savefig(pic_name_and_path + '.svg', format='svg')
                # plt.show()  #show完之后再关闭相当于清空了所有画板上的数据



                ############################################################


                # 计算切线的函数
                def tangent_line(x0, y0, k):
                    l_of_x=np.sqrt(1/(1+k*k))
                    xs = np.linspace(x0 - l_of_x/2, x0 + l_of_x/2, 100)
                    ys = y0 + k * (xs - x0)
                    return xs, ys


                # 计算斜率的函数,也就是在x0处求导
                def slope(num):
                    if ((xdata[num+1] - xdata[num])!=0):
                        k = (ydata[num+1] - ydata[num]) / (xdata[num+1] - xdata[num])
                    else:
                        k = (ydata[num+1] - ydata[num]) / 0.001
                    return k



                # 更新函数
                def updata(n):
                    # 绘制曲线上的切点
                    point_ani = plt.plot(xdata[start], ydata[start], 'r', alpha=0.4, marker='o')[0]

                    # 绘制x、y的坐标标识
                    ntext_ani = plt.text(xdata[start], ydata[start] + ticker_y, '', fontsize=12)
                    xtext_ani = plt.text(xdata[start], ydata[start] + ticker_y * 1.4, '', fontsize=12)
                    ytext_ani = plt.text(xdata[start], ydata[start] + ticker_y * 1.8, '', fontsize=12)
                    ktext_ani = plt.text(xdata[start], ydata[start] + ticker_y * 2.2, '', fontsize=12)

                    num=n+start
                    k = slope(num)
                    xs, ys = tangent_line(xdata[num], ydata[num], k)
                    tangent_ani.set_data(xs, ys)
                    point_ani.set_data(xdata[num], ydata[num])
                    ntext_ani.set_text('k=%.3f' % num)
                    xtext_ani.set_text('x=%.3f' % xdata[num])
                    ytext_ani.set_text('y=%.3f' % ydata[num])
                    ktext_ani.set_text('k=%.3f' % k)
                    return [point_ani, ntext_ani, xtext_ani, ytext_ani, tangent_ani, k]

                if animation_flag==1:
                    # 绘制切线
                    k = slope(start)
                    xs, ys = tangent_line(xdata[start], ydata[start], k)
                    tangent_ani = plt.plot(xs, ys, c='r', alpha=0.8)[0]

                    interval=90 #单位是ms
                    ani = animation.FuncAnimation(fig=fig, func=updata, frames=(end-start), interval=90)
                    # init 的作用是绘制下一帧画面前清空画布窗口的当前画面。update 的作用是绘制每帧画面
                    # frames 参数可以取值 iterable, int, generator 或者 None。如果取整数 n，相当于给参数赋值 range(n)
                    # frames 会在 interval 时间内迭代一次，然后将值传给 func，直至 frames 迭代完毕。
                    # ani.save("animation.mp4", fps=int(1000/interval), writer="ffmpeg")
                    ani.save(gif_name_and_path, fps=int(1000/interval), writer="imagemagick")
                if(show):
                    plt.show()

                # y1data = data.loc[:, '列名1'] #多条曲线的y值 参数名为csv的列名
                # y2data = data.loc[:, 'central_point_y']
                # y3data = data.loc[:, '列名3']


                # color可自定义折线颜色，marker可自定义点形状，label为折线标注
                # plt.plot(xdata, y1data, color='r', label=u'1路')#点标记,红色
                # plt.plot(xdata, y2data, color='b', label=u'2路')#星形标记,蓝色
                # plt.plot(xdata, y3data, color='g', label=u'3路')#上三角标记,绿色


                # # 绘制曲线最大最小值
                # # max1_indx=np.argmax(y1data)#找出曲线1的最大值下标
                # # min1_indx=np.argmin(y1data)#最小值下标
                # # plt.plot(max1_indx,y1data[max1_indx],'ks')
                # # plt.plot(min1_indx,y1data[min1_indx],'gs')
                #
                # max2_indx = np.argmax(y2data)  # 找出曲线2的最大值下标
                # min2_indx = np.argmin(y2data)  # 最小值下标
                # plt.plot(max2_indx, y2data[max2_indx], 'ks')
                # plt.plot(min2_indx, y2data[min2_indx], 'gs')
                #
                # max3_indx = np.argmax(y3data)  # 找出曲线3的最大值下标
                # min3_indx = np.argmin(y3data)  # 最小值下标
                # plt.plot(max3_indx, y3data[max3_indx], 'ks')
                # plt.plot(min3_indx, y3data[min3_indx], 'gs')
                #
                #
                # plt.text(0,y1data[max1_indx]+360, "1路最大值:"+str(y1data[max1_indx])+"最小值:"+str(y1data[min1_indx]), size = 10, alpha = 1)
                # plt.text(100,y2data[max2_indx]-0.5, "2路最大值:"+str(y2data[max2_indx])+"最小值:"+str(y2data[min2_indx]), size = 10, alpha = 1)
                # plt.text(100,y3data[max2_indx]-0.6, "3路最大值:"+str(y3data[max3_indx])+"最小值:"+str(y3data[min3_indx]), size = 10, alpha = 1)
                #
                # plt.axhline(y=1000.24, ls=":", c="red")  # 添加水平辅助线
                #
                # plt.savefig(r'E:\data\manual\processData\lxk\1.jpg')#保存图片 如果要保存图像，保存代码一定要写在显示图像之前，否则会出现保存空白图像的情况
                # plt.show()
                #
                # def get_rmse(list_y,predict): #求均方根函数，list_y是估计值列表，predict是预测值
                #     sum = 0
                #     N = len(list_y)
                #     for i in range(N):
                #         sum+=(list_y[i]-predict)**2
                #     rmse = math.sqrt(sum/N)
                #     return rmse
        except Exception as e:
            print(name_full+'处理出错')
            print(e)
            traceback.print_exc()
            name_wrong=name_wrong+'  '+name_full
            flag_error='有文件处理出问题'+name_wrong
            pass
    print(flag_error)
