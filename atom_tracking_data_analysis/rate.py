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
from mpl_toolkits.mplot3d import Axes3D  # Importing 3D functionality
from matplotlib.ticker import MaxNLocator

list_dir_sub=[
    'rate_test/1024rate_test1215','rate_test/1024rate_test1218','rate_test/1024rate_test1219'
    ,'rate_test/1024rate_test1220','rate_test/1024rate_test1220-1','rate_test/1024rate_test1220-2'
    ,'rate_test/1024rate_test1220-3'
    ,'rate_test/1024rate_test1222','rate_test/1024rate_test1222-1'
    ,'rate_test/1024rate_test1224'
              ]
'''
C:/Users/14170/Desktop/python/atom_tracking_data_analysis/data/vary_gradient_radius/rate_test/1024rate_test1220-1/
gradient_radius_002c_002c_rate_test0.png 收敛
gradient_radius_0010_0004_rate_test0.png 震荡
gradient_radius_0034_0009_rate_test0.png 收敛


C:/Users/14170/Desktop/python/atom_tracking_data_analysis/data/vary_gradient_radius/rate_test/1024rate_test1220-3/ 
gradient_radius_0058_0034_rate_test0.png 完美收敛
gradient_radius_0069_0044_rate_test0.png 收敛
gradient_radius_0097_0064_rate_test0.png 收敛
gradient_radius_0080_000c_rate_test0.png 缓慢收敛
gradient_radius_0080_0004_rate_test0.png 缓慢收敛
gradient_radius_0080_0034_rate_test0.png 缓慢收敛
gradient_radius_0058_0048_rate_test0.png 凸起收敛
gradient_radius_00a8_0008_rate_test0.png
gradient_radius_00a8_0070_rate_test0.png 震荡
gradient_radius_0040_000c_rate_test0.png 震荡
gradient_radius_0058_002c_rate_test0.png 震荡
gradient_radius_0069_0034_rate_test0.png 震荡收敛


C:/Users/14170/Desktop/python/atom_tracking_data_analysis/data/vary_gradient_radius/rate_test/1024rate_test1222/
gradient_radius_00d6_0016_rate_test0.png
gradient_radius_0004_0008_rate_test0.png 收敛
gradient_radius_005e_0010_rate_test0.png 震荡
gradient_radius_0013_0067_rate_test0.png 收敛（步长大于扫描半径）
gradient_radius_0022_0028_rate_test0.png 收敛震荡
gradient_radius_0040_0028_rate_test0.png 收敛震荡


C:/Users/14170/Desktop/python/atom_tracking_data_analysis/data/vary_gradient_radius/rate_test/1024rate_test1222-1/
gradient_radius_01d5_001c_rate_test0.png 收敛
gradient_radius_01f3_000a_rate_test0.png 收敛
gradient_radius_004f_0010_rate_test0.png 收敛
gradient_radius_004f_0022_rate_test1.png 震荡收敛
gradient_radius_005e_0028_rate_test0.png 收敛震荡
gradient_radius_007c_0016_rate_test0.png 收敛
gradient_radius_008b_003a_rate_test1.png 震荡
gradient_radius_008b_0022_rate_test0.png 收敛
gradient_radius_0013_004f_rate_test1.png 收敛
gradient_radius_015d_003a_rate_test0.png 收敛
gradient_radius_0022_0004_rate_test1.png 收敛
gradient_radius_0022_0022_rate_test1.png
gradient_radius_0022_002e_rate_test1.png


C:/Users/14170/Desktop/python/atom_tracking_data_analysis/data/vary_gradient_radius/rate_test/1024rate_test1224/ 
gradient_radius_0004_0008_rate_test0.png 收敛震荡
gradient_radius_0031_0034_rate_test0.png
gradient_radius_004f_000a_rate_test0.png 收敛
gradient_radius_0013_0013_rate_test0.png 震荡
gradient_radius_0022_0028_rate_test0.png 收敛
gradient_radius_0031_0034_rate_test0.png 收敛震荡
'''

animation_flag  =0
start_init      =0
end_init        =0  #所想绘制的数据点的绝对index
show            =0
save_pic_type='svg' # 0:png 1:svg  eps
point_minute    =30
#################待处理的文件夹及文件#########################
file_choose=1
name_list=['record_at_gradient_helix_drift0024_0010.csv','record_at_gradient_helix_drift00a4_0008.csv']
re_file='*rec*.csv'#相应文件的正则表达式
mode_algorithm =1
mode_scan      =0
mode_measure   =0
# dir_sub='rate_test/1024rate_test1220-3'

algorithm='';scan='';measure='';
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
name_dir = 'vary_'+algorithm+'_'+scan
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
my_cmap1_r = mpl.colors.LinearSegmentedColormap.from_list(name='mycmp1_r', colors=[endcolor, midcolor, startcolor])
mpl.colormaps.register(name='mycmp1_r', cmap=my_cmap1_r)
# 全局画图配置
plt.rcParams['font.sans-serif'] = ['Arial']
# 如果要显示中文字体,则在此处设为：SimHei 由于存在多个Arial字体，暂时还没有找到改变映射的方法
# Arial Rounded MT Bold - C:\Windows\Fonts\ARLRDBD.TTF 最后一个被映射的，但在adobe中没有该字体无法识别
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['font.family'] = 'sans-serif'  # 设置字体格式
plt.rcParams['font.size'] = 12
from matplotlib import font_manager
for font in font_manager.fontManager.ttflist:
    # 查看字体名以及对应的字体文件名
    print(font.name, '-', font.fname)
myfont = mpl.font_manager.FontProperties(fname=r"c:\windows\fonts\arial.ttf", size=14)
# fig,ax=plt.subplots()
# ax.set_title(u'style', fontproperties = myfont)
# plt.show()

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# # 字体文件的路径
# font_path = r"C:\Windows\Fonts\arial.ttf"
# # 配置全局字体设置
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.serif'] = f"CustomFontName:file://{font_path}"
# Times New Roman - C:\Windows\Fonts\timesbd.ttf
# Arial - C:\Windows\Fonts\arial.ttf

##############################################################################################
#######################  main   ################################
##############################################################################################
for index,dir_sub in enumerate(list_dir_sub):
    #############################################################################

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





    flag_error='全部处理成功'
    name_wrong=''

    name_files=''
    if(file_choose==1):
        name_files = name_all#遍历全部文件
    elif(file_choose==2):
        name_files = name_list
    elif(file_choose==3):
        name_files = glob.glob(re_file)


    # gs = GridSpec(len(name_files), 1, figure=fig)
    for i, name_full in enumerate(name_files):
        try:
            print('待处理的数据' + name_full)
            portion = os.path.splitext(name_full)
            name = portion[0]
            # 如果后缀是.txt
            if ((portion[1] == ".csv")&(portion[0][-1]!='z')):
                print('正在处理 ',path,name_full)
                # input
                data_name_and_path = name + '.csv'
                # output
                pic_name_and_path = path_pic + name
                gif_name_and_path = path_pic + name + '.gif'


                def function(x):  # input is orignal DAC data string,2 LSB is oversample,4 MSB is control bits
                    x_bin = myfunc.hex2bin(x, 26)   # x='3ffffff' x_bin='0b11111111111111111111111111'
                    x_bin='0b'+x_bin[6:26]          #             x_bin='0b11111111111111111111'
                    right_shift_num = 0             # 符号位要在n byte的最左边
                    hva=4
                    x_shift = x_bin + right_shift_num * '0'
                    x_sign = myfunc.signed_bin2dec(x_shift)
                    voltage = (x_sign /  (2 ** right_shift_num)) * (20 / (2 ** 20)) * hva
                    movment = voltage * 20  # nm/V
                    return movment  # DAC output transform to movement of tip

                def function_z(x):  # input is 22bits orignal DAC data,2 LSB is oversample
                    x_bin = myfunc.hex2bin(x, 26)   # x='3ffffff' x_bin='0b11111111111111111111111111'
                    x_bin='0b'+x_bin[6:26]          #             x_bin='0b11111111111111111111'
                    right_shift_num = 0             # 符号位要在n byte的最左边
                    hva=4
                    x_shift = x_bin + right_shift_num * '0'
                    x_sign = myfunc.signed_bin2dec(x_shift)
                    voltage = (x_sign /  (2 ** right_shift_num)) * (20 / (2 ** 20)) * hva
                    movment = voltage * 3  # nm/V
                    return movment  # DAC output transform to movement of tip

                dac_20bits = '00004'  # '6d3'
                dac_20bits_bin = myfunc.hex2bin(dac_20bits, 20)
                dac_22bits_bin = dac_20bits_bin + 2 * '0'
                dac_22bits = myfunc.bin2hex(dac_22bits_bin, 6)
                print("dac_22bits is:", dac_22bits, "\ndac_movement_value is: ", function(dac_22bits), "nm")
                print("dac_22bits_xy is:0001", "\ndac_movement_value is: ", function('0004'), "nm")
                print("dac_22bits_z  is:0001", "\ndac_movement_value is: ", function_z('0004'), "nm")

                # 读文件
                data = pd.read_csv(data_name_and_path)
                # print(data.dtypes)
                data['x'] = data['x'].astype(str)
                data['y'] = data['y'].astype(str)
                data['z'] = data['z'].astype(str)
                # print(data.dtypes)
                # data['x'] = data['x'].apply(int, base=16).map(function)
                # data['y'] = data['y'].apply(int, base=16).map(function)
                data['x'] = data['x'].map(function)
                data['y'] = data['y'].map(function)
                data['z'] = data['z'].map(function_z)
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
                zdata = data.loc[start:end, 'z']
                print(type(zdata),zdata)

                xdata_max=max(xdata);xdata_min=min(xdata)
                ydata_max=max(ydata);ydata_min=min(ydata)
                zdata_max=max(zdata);zdata_min=min(zdata)
                colors = np.arange(0,1,(1-0)/(end-start+1))#从0开始太淡了
                # 如果想得到小数数列的话可以用numpy中的arange函数，
                # 自带的range函数只能得到整数类型的序列（注意当需要小数序列时用该函数会报错）。
                rgb = mpl.colormaps['Blues'](colors)[np.newaxis, :, :3]
                # print('colors= ',colors,'rgb= ',rgb,'type(rgb)= ',type(rgb))
                lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
                # print('lab= ',lab, 'type(lab)= ',type(lab))



                ###########################################################################################
                # 定义两个拟合函数
                def func_y(x, ay, by, cy):
                    return ay * x ** 2 + by * x + cy


                def func_z(x, az, bz, cz):
                    return az * x ** 2 + bz * x + cz


                # 拟合 y 和 z 作为 x 的函数
                popt_y, _ = curve_fit(func_y, xdata, ydata)
                popt_z, _ = curve_fit(func_z, xdata, zdata)
                print('popt_y=', popt_y, 'popt_z=', popt_z, '\n')
                # 使用拟合参数生成 y 和 z 的数据
                y_fit = func_y(xdata, *popt_y)
                z_fit = func_z(xdata, *popt_z)

                ########################################################################################################
                # 初始化画布
                # fig= plt.subplots(figsize=(11, 8))
                # gs = mpl.gridspec.GridSpec(1, 4, width_ratios=[3, 1])  # 创建4个子图，宽度比例为3:1
                fig = plt.figure(figsize=(4.7, 7.1))
                ax1 = fig.add_axes([0.02, 0.36, 0.7, 0.64], projection='3d')  # 添加一个新的子图
                # 第一个值0.85 表示子图的左边界距离整个画布左边界的相对位置，即子图的x起始位置在整个画布宽度的85%处。
                # 第二个值0.1表示子图的下边界距离整个画布下边界的相对位置，即子图的y起始位置在整个画布高度的10%处。
                # 第三个值0.15表示子图的宽度相对于整个画布宽度的比例，即子图宽度为整个画布宽度的15%。
                # 第四个值0.8表示子图的高度相对于整个画布高度的比例，即子图高度为整个画布高度的80%。
                ###############################################################################

                # Create 3D scatter plot
                # gs.update(left=0.1, right=0.4, bottom=0.5, top=0.9, wspace=0.5, hspace=0.5)
                # 绘制原始数据点
                colors = np.arange(len(xdata)) # np.squeeze(xdata)
                ax1.scatter(xdata, ydata,zdata, c=colors, cmap=mpl.colormaps.get_cmap('mycmp1'),alpha = 8/10,s=16
                    , label = 'Real trace'
                    #         'x_fit :%.3f pm/min \n'
                    #         'y_fit :%.3f pm/min \n'
                    #         'x_max :%.3f pm/min \n'
                    #         'y_max :%.3f pm/min \n'
                    #         'zdata_max :%.2f'
                    #         % (
                    # 1000*(xdata[start] - xdata[end]) / ((end - start) / point_minute),
                    # 1000*(y_fit[start] - y_fit[end]) / ((end - start) / point_minute),
                    # 1000*(xdata_max - xdata_min) / ((end - start) / point_minute),
                    # 1000*(ydata_max - ydata_min) / ((end - start) / point_minute),
                    # zdata_max )
                            )
                # 绘制拟合曲线
                ax1.plot(xdata, y_fit, z_fit, label='Fitted curve', color='r')
                plt.legend(loc='upper left', bbox_to_anchor=(0.5, 1.0), borderaxespad=0, frameon=False)#(0.37, 0.8)  (0.4, 0.3)


                size_small=14
                size_big=20
                # plt.title(name+"\'thermal drift", fontsize=16)  # 曲线标题
                plt.xlabel(u'X/nm', fontsize=size_big,labelpad=15)
                # 设置x轴标签旋转角度和字体大小
                plt.xticks(rotation=0, fontsize=size_small)
                plt.ylabel(u'Y/nm', fontsize=size_big, labelpad=20)
                plt.yticks(rotation=0, fontsize=size_small)
                # 设置Z轴标签并调整字体大小
                ax1.set_zlabel('Z/nm', fontsize=size_big, labelpad=8, rotation=90)  # 可以根据需要调整字体大小
                # 调整Z轴刻度值的字体大小
                ax1.tick_params(axis='z', labelsize=size_small)  # 可以根据需要调整字体大小
                ticker_x=abs((xdata_max - xdata_min)/5)
                ticker_y=abs((ydata_max - ydata_min)/5)
                # 设置X、Y、Z轴的刻度数量
                ax1.xaxis.set_major_locator(MaxNLocator(4))  # X轴最多显示5个刻度
                ax1.yaxis.set_major_locator(MaxNLocator(4))  # Y轴最多显示6个刻度
                ax1.zaxis.set_major_locator(MaxNLocator(4))  # Z轴最多显示7个刻度
                # 创建一个标准化器和一个颜色映射对象
                # norm = mcolors.Normalize(vmin=0, vmax=10)
                norm = mcolors.Normalize(vmin=0, vmax=len(zdata))
                im = cm.ScalarMappable(norm=norm, cmap=mpl.colormaps.get_cmap('mycmp1'))
                cax = fig.add_axes([0.87, 0.5, 0.02, 0.35])  # [left, bottom, width, height] in figure coordinate
                # cb = fig.colorbar(im, cax=cax, orientation='vertical', ticks=np.linspace(0, 10, 11))
                cb = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[int (tick) for tick in np.linspace(0, len(zdata), 2)])
                # 旋转colorbar的刻度标签
                cb.ax.set_yticklabels(cb.ax.get_yticklabels(), rotation=90)
                # 设置colorbar的标签和样式
                cb.set_label('Index', fontsize=size_big, labelpad=0)  # $\mathrm{(^oC)}$
                cb.outline.set_linewidth(0.1)
                # 设置colorbar的刻度标签字体大小
                cb.ax.tick_params(labelsize=size_small)  # 这里将字体大小设置为10
                # 手动调整视角
                elev = 36   # 举例，您可以根据需要调整这个值
                azim = 130  # 举例，您可以根据需要调整这个值
                ax1.view_init(elev=elev, azim=azim)


                ###################################
                # 绘制zdata随索引的变化
                ax2 = fig.add_axes([0.15, 0.1, 0.8, 0.25])  # 添加一个新的子图
                ax2.grid(True)
                ax2.plot(range(len(zdata)), zdata, label='Z-real', color='b')
                ax2.set_xlabel('Index', fontsize=size_big)
                ax2.set_ylabel('Z/nm', fontsize=size_big)
                plt.xticks(rotation=0, fontsize=size_small)
                plt.yticks(rotation=90, fontsize=size_small)
                ax2.xaxis.set_major_locator(MaxNLocator(4))  # X轴最多显示5个刻度
                ax2.yaxis.set_major_locator(MaxNLocator(2))  # Y轴最多显示6个刻度
                # ax2.set_title(name + '\'s Zdata')
                # 多项式拟合函数
                def poly_fit(x, y, degree):
                    coeffs = np.polyfit(x, y, degree)
                    return np.poly1d(coeffs)
                # 应用拟合函数
                fit_degree = 10  # 你可以根据需要更改这个拟合次数
                z_poly_fit = poly_fit(range(len(zdata)), zdata, fit_degree)

                # 在ax2中绘制拟合曲线
                ax2.plot(range(len(zdata)), z_poly_fit(range(len(zdata))), color='black', linewidth=2,
                         label=f'Z-fit') # {fit_degree}-degree
                # Get the current y-axis limits
                y_min, y_max = ax2.get_ylim()
                # Extend the range by a certain percentage of the current range, e.g., 10%
                y_range_extension = 0.12 * (y_max - y_min)
                # Set new y-axis limits
                ax2.set_ylim(y_min - y_range_extension, y_max )
                # plt.subplots_adjust(left=0.05, right=0.85, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
                ax2.legend(loc='lower right', borderaxespad=0,fontsize=14, frameon=False) # bbox_to_anchor=(0.8, 0.2)
                # plt.tight_layout()
                # plt.savefig(pic_name_and_path + '.png') #
                plt.savefig(pic_name_and_path + '.' + save_pic_type, format=save_pic_type)
                ################################################################################
                if(show):
                    plt.show()
                    # 打印视角
                    print(f"当前视角: 仰角 {ax1.elev}, 方位角 {ax1.azim}")
                    # 获取并打印调整后的图形尺寸
                    fig_size = fig.get_size_inches()
                    print("图形的尺寸为: 宽度 {:.2f}英寸, 高度 {:.2f}英寸".format(fig_size[0], fig_size[1]))


        except Exception as e:
            print(name_full+'处理出错')
            print(e)
            traceback.print_exc()
            name_wrong=name_wrong+'  '+name_full
            flag_error='有文件处理出问题'+name_wrong
            pass
    print(flag_error)