# -*- coding: utf-8 -*-
"""

author: Filliam
Created on 11.06.2023

Description:
============
those code based on Felix Wählisch's work

Disclaimer:
===========
I am not responsible to any havoc, malevolence or inconvenience that happens to
you or your data while using this script (:

License:
========
"THE BEER-WARE LICENSE":
As long as you retain this notice you can do whatever you want with this stuff.
If we meet some day, and you think this stuff is worth it, you can buy me
a beer in return.
    - Felix Wählisch

"""

import numpy as np
import pylab as P
import matplotlib.pyplot as plt
import os
import re
import string
import struct
from scipy import optimize
import csv
import process_method
from scipy.ndimage import zoom

class NanonisAFM:
    ''' data contains:
        fname           filename
        infos           dictionary with all header entries
        datachannels    array of the datachannels in datachannel class format
    '''

    def __init__(self, fname=''):
        ''' creates empty class and asks for file to process if not given '''
        self.signals = []
        self.infos = {}
        self.fname = ''
        if fname == '':
            # your custom ask for file
            print("init(): please program me!")
        elif fname == '':
            print("selection aborted")
            return
        else:
            self.fname = fname
            self.basename = os.path.splitext(fname)[0]
            fname_array=os.path.split(self.fname)
            print("file: ", fname_array[1])
            self.readin()




    def _checkfile(self):
        '''inherit with some reality-checks that verify a good filename. It is recommended to run this after selecting self.fname'''
        if self.fname.endswith('.sxm'):
            return True
        else:
            print("wrong file ending (not .sxm)", self.fname)
            return False

    def readin(self):
        if not self._checkfile():
            return -1
        self._read_header()
        self._read_body()

    def _read_header(self):
        ''' reads the header and adds to info dictionary - ready for further parsing as needed'''
        header_ended = False
        fhandle = open(self.fname, 'r', encoding='gb18030', errors='ignore')
        caption = re.compile(':*:')
        key = ''
        contents = ''
        while not header_ended:
            line = fhandle.readline()
            if line == ":SCANIT_END:\n":  ## check for end of header
                header_ended = True
                self.infos[key] = contents
                ## two blank lines
                fhandle.readline();
                fhandle.readline()
            else:
                if caption.match(line) != None:  ## if it is a caption
                    if key != '':  # avoid 1st run problems
                        self.infos[key] = contents
                    key = line[1:-2]  # set new name
                    contents = ''  # reset contents
                else:  # if not caption, it is content
                    contents += (line)
        fhandle.close()
        print('header:',self.infos,'\n')

    def _read_body(self):
        '''The binary data begins after the header and is introduced by the (hex) code \1A\04.
        According to SCANIT_TYPE the data is encoded in 4 byte big endian floats.
        The channels are stored one after the other, forward scan followed by backward scan.
        The data is stored chronologically as it is recorded.
        On an up-scan, the first point corresponds to the lower left corner
        of the scanfield (forward scan). On a down-scan, it is the upper
        left corner of the scanfield.
        Hence, backward scan data start on the right side of the scanfield.'''
        ## extract channes to be read in
        data_info = self.infos['DATA_INFO']
        print('data_info:\n',data_info)
        lines = str.split(data_info, '\n')
        lines.pop(
            0)  # headers: Channel    Name      Unit        Direction                Calibration               Offset
        names = []
        units = []
        for line in lines:
            entries = str.split(line)
            if len(entries) > 1:
                names.append(entries[1])
                units.append(entries[2])
                if entries[3] != 'both':
                    print("warning, only one direction recorded, expect a crash :D", entries)
        ## extract lines, pixels
        # xPixels = int(self.infos['Scan>pixels/line'])
        # yPixels = int(self.infos['Scan>lines'])
        xPixels, yPixels = str.split(self.infos['SCAN_PIXELS'])
        xPixels = int(xPixels)
        yPixels = int(yPixels)
        print('xPixels:',xPixels,'yPixels:', yPixels)
        ## find position in file
        fhandle = open(self.fname, 'rb')  # read binary
        read_all = fhandle.read()
        offset = read_all.find('\x1A\x04'.encode())
        print('found start at', offset)
        fhandle.seek(offset + 2)  # data start 2 bytes afterwards
        ## read in data
        # SCANIT_TYPE 描述数据是如何存储的
        # 大于号：大端存储，小于号：小端存储；float：浮点数 int:整型
        fmt_type,fmt_post=str.split(self.infos['SCANIT_TYPE'])
        fmt_type_map={'FLOAT':'f','INT':'i'}
        fmt_post_map={'MSBFIRST':'>','LSBFIRST':'<'}
        fmt=fmt_post_map[fmt_post]+fmt_type_map[fmt_type]
        ItemSize = struct.calcsize(fmt) # 该文本的存储的字的字节数
        print('ItemSize:',ItemSize)
        for i in range(len(names) * 2):  # fwd+bwd
            if i % 2 == 0:
                direction = '_fwd'
            else:
                direction = '_bwd'
            bindata = fhandle.read(ItemSize * xPixels * yPixels)
            data = np.zeros(xPixels * yPixels)
            for j in range(xPixels * yPixels):
                data[j] = struct.unpack(fmt, bindata[(j * ItemSize): (j * ItemSize + ItemSize)])[0]
            # python的reshape是先行后列
            data = data.reshape(yPixels, xPixels)
            # data = np.rot90(data)
            if direction == '_bwd':
                data = np.fliplr(data)#flip left<->right
            scan_dir = self.infos['SCAN_DIR'].strip()  # 去除字符串两端的空白字符
            if scan_dir=='down' :
                data = data[::-1]
                print("down scan")
            channel = Datachannel(name=names[i // 2] + direction, data=data, unit=units[i // 2])
            print(channel.name, channel.unit, channel.data.shape)
            self.signals.append(channel)
        fhandle.close()
        # 在函数的末尾添加保存CSV的调用
        if (save_csv):
            self.save_data_to_csv()

    def save_data_to_csv(self):
        """将所有通道的数据保存到一个CSV文件中"""
        csv_filename = self.basename + '.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 创建列标题
            headers = ['x', 'y']
            for signal in self.signals:
                headers.append(signal.name)
            writer.writerow(headers)

            # 遍历所有数据点
            for y in range(self.signals[0].data.shape[0]):
                for x in range(self.signals[0].data.shape[1]):
                    row = [x, y]
                    # 遍历每个通道，获取对应位置的数据
                    for signal in self.signals:
                        row.append(signal.data[y, x])
                    writer.writerow(row)



    def create_img(self, nametag, clim=(None, None)):
        '''puts out images of signals whose name contains the nametag.
        adjust your color bar by using clim(lower, upper)'''
        x_len, y_len = str.split(self.infos['SCAN_RANGE'])#以空格或\n为分隔符，也可以自行设置
        x_len = float(x_len)
        y_len = float(y_len)
        # if you change to nm, also change the labels further below (;
        x_len *= 1.0e9  # um
        y_len *= 1.0e9  # um
        print(x_len,y_len)
        for i in self.signals:
            if nametag in i.name:
                z=i.data
                ###################data process################################################
                if i.unit != 'V':
                    z = z * 1.0e9
                    i.unit = 'n' + i.unit
                print("z.shape", z.shape,"z", z)
                # 删除含有 NaN 的行
                # z_clean = z[~np.isnan(z).any(axis=0)]
                # print("z_clean.shape", z_clean.shape)
                # 分离数据
                rows_with_nan = np.isnan(z).any(axis=1)
                z_no_nan = z[~rows_with_nan]  # 不含 NaN 的部分
                z_with_nan = z[rows_with_nan]  # 含有 NaN 的部分
                print("z_no_nan.shape", z_no_nan.shape,'z_with_nan.shape',z_with_nan.shape)
                # 如果您希望删除列，可以将axis = 1改为axis = 0。
                ###################data process################################################
                # match process_choice:
                #     case 1:
                #         # 对所有数据进行平面处理
                #         z_processed = process_method.subtract_plane(z_clean)
                #     case 2:
                #         # 对每一行进行二次曲线拟合并减去拟合曲线
                #         z_processed = process_method.subtract_curve(z_clean)
                #     case 3:
                #         z_processed = process_method.invert(z_clean)
                #     case 4:
                #         z_processed = process_method.sharpen(z_clean, alpha=1.0)
                #     case _:
                #         print("Something else")
                # 检查 z_no_nan 是否为空
                if z_no_nan.size > 0:
                    # 如果不为空，继续处理
                    # 增加对比度
                    # 计算数据的5%和95%分位数
                    lower_percentile, upper_percentile = np.percentile(z_no_nan, [0, 100])
                    # 将数据缩放到这个范围内
                    z_no_nan = np.clip((z_no_nan - lower_percentile) / (upper_percentile - lower_percentile), 0, 1)
                    if (en_subtract_plane):
                        # 对所有数据进行平面处理
                        z_processe1 = process_method.subtract_plane(z_no_nan)
                    else:z_processe1=z_no_nan
                    if (en_subtract_curve):
                        # 对每一行进行二次曲线拟合并减去拟合曲线
                        z_processe2 = process_method.subtract_curve(z_processe1)
                    else:z_processe2=z_processe1
                    if (en_invert):
                        z_processe3 = process_method.invert(z_processe2)
                    else:z_processe3=z_processe2
                    if (en_sharpen):
                        z_processe4 = process_method.sharpen(z_processe3, alpha=1.0)
                    else:z_processe4=z_processe3
                    if (en_histogram_equalization):
                        z_processe5 = process_method.histogram_equalization(z_processe4)
                    else:z_processe5=z_processe4
                    z_no_nan_processed =z_processe5
                else:
                    z_no_nan_processed =z_no_nan

                # 处理含有 NaN 的部分 - 转换为全黑色
                # z_with_nan_processed = np.zeros_like(z_with_nan) # 将空白变成0值所对应的颜色，灰度图是为黑色
                z_with_nan_processed = z_with_nan #仍然保持空白



                # 合并图像
                z_processed = np.vstack((z_no_nan_processed, z_with_nan_processed))
                print("z_processed.shape", z_processed.shape)
                dpi = 100  # Set the desired DPI
                fig_size_x = z_processed.shape[0] / dpi  # Calculate the figure size for allocated pixels
                fig_size_y = z_processed.shape[1] / dpi  # Calculate the figure size for allocated pixels
                plt.figure(figsize=(fig_size_x, fig_size_y), dpi=dpi)  # Set figure size


                # 绘制处理后的图像
                print("create_img(): creating", i.name, "image...")
                plt.imshow(z_processed, origin="lower", cmap=plt.cm.YlOrBr_r, aspect='equal')
                ax = plt.subplot(111)
                if fig_ax:
                    # ticker adjustment
                    (yi, xi) = z.shape
                    # get old x-labels, create new ones
                    x_ticks = np.int_(np.round(np.linspace(0, xi - 1, len(ax.axes.get_xticklabels()))))
                    x_tlabels = np.round(np.linspace(0, x_len, len(ax.axes.get_xticklabels())), decimals=3)
                    ax.axes.set_xticks(x_ticks)
                    ax.axes.set_xticklabels(x_tlabels)
                    plt.xlim(0, xi - 1)  # plots from 0 to p-1, so only show that
                    plt.xlabel("X [nm]")
                    # get old y-labels, create new ones, note reverse axis ticks
                    y_ticks = np.int_(np.round(np.linspace(0, yi - 1, len(ax.axes.get_yticklabels()))))
                    y_tlabels = np.round(np.linspace(0, y_len, len(ax.axes.get_yticklabels())), decimals=2)
                    ax.axes.set_yticks(y_ticks)
                    ax.axes.set_yticklabels(y_tlabels)
                    plt.ylim(0, yi - 1)  # plots from 0 to p-1, so only show that
                    plt.ylabel("Y [nm]")
                    if clim != (None, None):
                        plt.clim(clim[0], clim[1])
                    bar = plt.colorbar(shrink=0.7)
                    bar.set_label(i.name + ' [' + i.unit + ']')
                    plt.title(os.path.split(self.fname)[1][:-4])
                    plt.draw()
                else:
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)  # 调整子图间距
                    plt.axis('off')  # Turn off the axes
                current_directory = os.getcwd()
                print("当前路径:", current_directory)
                print(dir_figs+'/'+self.fname[:-4] + '_' + i.name + '.png')
                # Check if the folder exists, and create it if it doesn't
                if (save_fig):
                    plt.savefig(dir_figs+'/'+self.fname[:-4] + '_' + i.name + '.png',
                                dpi=dpi,transparent=True, bbox_inches='tight', pad_inches=0)
                    # 官方解释：bbox_inches控制要保存的图像的边界框(boundingbox)。
                    # 默认情况下，它的值为'tight'，这意味着Matplotlib会尽量剪裁图像周围的空白区域，
                    # 使图像尽可能地紧凑，以包含所有图形元素。
                    # 使用方式：plt.savefig('output.png', bbox_inches='tight')
                    # 官方解释：pad_inches控制图像保存时在边界框周围添加的填充。
                    # 默认情况下，它的值为0.1，表示在边界框周围添加0.1英寸的填充。
                    # 使用方式：plt.savefig('output.png', pad_inches=0)
                if(show):
                    plt.show()
                plt.close()

    def substract_1Dfit(self, deg=1):
        '''substracts a line by line fit from all data
        degree = 0 offset fit
        degree = 1 linear fit
        degree = 2 ...
        '''
        Fit = fit()
        for i in self.signals:
            data = i.data
            res = Fit.poly_line_by_line(data, deg, axis=1)
            i.data = data - res
        return 0

    def substract_2Dfit(self, deg=2):
        '''substracts something like a 2D fit from all data
        degree: 1- plane substract
                2- parabolic substract'''
        if deg not in range(1, 3, 1):  # goes to n-1
            print("substract_2Dfit(): unknown degree of fit, abort...")
            return -1
        Fit = fit()
        for i in self.signals:
            data = i.data
            # fit parameters initial values
            if deg == 1:
                params = Fit.fitplane(data)
                fit_func = Fit.return_plane(params, data)
            if deg == 2:
                params = Fit.fitparabolic(data)
                fit_func = Fit.return_parabolic(params, data)
            i.data = data - fit_func
        return 0

    def friction_signal(self, ignore_large_img=True):
        ''' calculates Horiz._Deflection fwd - bwd / 2.0 and appends the Friction channel to the signals.
        Returns Friction.data. Ignores large images by default due to hysteresis.
        '''
        fwd = None
        bwd = None
        for i in self.signals:
            if i.name == "Horiz._Deflection_fwd":
                fwd = i.data
            if i.name == "Horiz._Deflection_bwd":
                bwd = i.data
                unit = i.unit  # unit of both channels is supposed to be same
            if i.name == 'Friction':
                print("friction_signal(): Friction channel already exists, aborting...")
                return -1
        if fwd == None or bwd == None:
            print("friction_signal(): could not find all signals needed, aborted.")
            return -1
        # ignore large images due to hysteresis
        x_len, y_len = str.split(self.infos['SCAN_RANGE'])
        x_len = float(x_len)
        y_len = float(y_len)
        x_len *= 1.0e6  # um
        y_len *= 1.0e6  # um
        if x_len > 30.0 or y_len > 30.0:
            if not ignore_large_img:
                print("friction_signal(): warning, the friction signal might be shadowed due to large scan range and hysteresis!")
            else:
                print("friction_signal(): friction signal is not created due to image size")
                return -1
        print("friction_signal(): creating Friction channel.")
        frict = Datachannel(data=(fwd - bwd) / 2.0, name='Friction', desc='horiz. deflection (fwd-bwd)/2', unit=unit)
        self.signals.append(frict)
        return frict.data


class Datachannel:
    '''data and their description...'''
    unit = ""  ## unit of the channel
    name = ""  ## name of the channel
    desc = ""  ## description / additional info
    data = np.array([])  ## data in the channel

    def __init__(self, unit="", name="", desc="", data=""):
        self.unit = str(unit)
        self.name = str(name)
        self.desc = str(desc)
        self.data = np.array(data)


class fit:
    """ 2D Fit Functions """

    def __init__(self):
        return

    ####################################
    """ taken from the Scipy Cookbook - gauss is not used from Nanonis AFM"""

    def gaussian(self, height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x, y: height * np.exp(
            -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)

    def gaussmoments(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X * data).sum() / total
        y = (Y * data).sum() / total
        col = data[:, int(y)]
        width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
        height = data.max()
        return height, x, y, width_x, width_y

    def fitgaussian(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = self.gaussmoments(data)
        errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)
        return p

    ####################################
    def parabolic(self, a0, a1, a2, b1, b2, x0, y0):
        '''could also do slope and plain - wow! used by substract_2Dfit'''
        return lambda x, y: a0 + a1 * (x - x0) + a2 * (x - x0) ** 2 + b1 * (y - y0) + b2 * (y - y0) ** 2

    def parabolicmoments(self, data):
        '''to be filled...'''
        a0 = abs(data).min()
        index = (data - a0).argmin()
        x, y = data.shape
        x0 = float(index / x)
        y0 = float(index % y)
        a1 = 0.0
        a2 = 0.0
        b1 = 0.0
        b2 = 0.0
        return a0, a1, a2, b1, b2, x0, y0

    def fitparabolic(self, data):
        params = self.parabolicmoments(data)
        errorfunction = lambda p: np.ravel(self.parabolic(*p)(*np.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)
        return p

    def return_parabolic(self, params, data):
        ''' returns an 2D array of the parabolic fit with the shape of data'''
        fit_data = self.parabolic(*params)
        return fit_data(*np.indices(data.shape))
        ####################################

    def plane(self, a0, a1, b1, x0, y0):
        return lambda x, y: a0 + a1 * (x - x0) + b1 * (y - y0)

    def planemoments(self, data):
        a0 = np.abs(data).min()
        index = (data - a0).argmin()
        x, y = data.shape
        x0 = float(index / x)
        y0 = float(index % y)
        a1 = 0.0
        b1 = 0.0
        return a0, a1, b1, x0, y0

    def fitplane(self, data):
        params = self.planemoments(data)
        errorfunction = lambda p: np.ravel(self.plane(*p)(*np.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)
        return p

    def return_plane(self, params, data):
        fit_data = self.plane(*params)
        return fit_data(*np.indices(data.shape))

    ####################################
    def poly_line_by_line(self, data, deg=1, axis=0):
        '''takes data, degree for polynomial line-by-line fitting,
        axis to fit along
        returns fitted surface'''
        if axis == 1:  # turn data around
            data = np.rot90(data)

        surface = np.zeros(data.shape)
        x = range(data.shape[1])
        for i in range(len(data)):
            p = np.polyfit(x, data[i], deg)
            surface[i] = np.polyval(p, x)

        if axis == 1:  # turn results back around
            surface = np.rot90(surface, k=3)
        return surface


## FINALLY - HERE IS YOUR EXAMPLE - Batch-Processing a whole folder
## and get some images automatically.
if __name__ == '__main__':
    base_dir=r"C:\Users\14170\Desktop\python\openSPM\sxm_process"
    # 存放待处理的文件夹列表
    dir_list=[r'\files_sxm\0408晚drift\au111-230408049'
        , r'\files_sxm\0406晚drift\au111-0406047'
        , r'\files_sxm\0406晚drift\au111-0406001'
        # , r'\files_sxm\202303 drift '
        # , r'\files_sxm\202306 drift '
              ]
    # sub_dir_files=r"\files_sxm\~"
    # 0408晚drift\au111-230408049
    # 0406晚drift\au111-0406047
    # 0406晚drift\au111-0406001
    # 202303 drift
    # 202306 drift
    for sub_dir_files in dir_list:
        sub_dir_figs=r"\fig"+sub_dir_files[10:]
        dir_files = base_dir+sub_dir_files
        dir_figs = base_dir+sub_dir_figs
        if not os.path.exists(dir_figs):
            os.makedirs(dir_figs)
        en_subtract_plane=1
        en_subtract_curve=0
        en_invert=0
        en_sharpen=0
        en_histogram_equalization=0
        save_csv=0
        fig_ax=0
        save_fig=1
        show=0

        os.chdir(dir_files)
        files = os.listdir(dir_files)
        print(files)
        for fname in files:
            if fname.endswith(".sxm"):
                a = NanonisAFM(fname)
                # a.friction_signal()
                # a.create_img(nametag='Friction')
                # a.substract_2Dfit(deg=1)
                # a.create_img(nametag="Horiz")
                # a.substract_2Dfit(deg=2)
                a.create_img(nametag="Z_fwd")
                # a.create_img(nametag="Z_bwd")
                print('\n')
        print("\033[32m""all done!""\033[0m")
        # print("\033[31m这是红色文字\033[0m")  # 红色
        # print("\033[32m这是绿色文字\033[0m")  # 绿色
        # 其中，\033[是引导序列，31m和32m分别代表不同的颜色，0m用于重置颜色


