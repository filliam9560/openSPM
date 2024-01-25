import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import drift_cv.drift_method as drift_method
import drift_cv.data_pic as data_pic


def process_images( start_num, end_num, save_dir,region_start,base_name):
    # 创建一个空的 1x2 的 NumPy 数组
    result = np.empty((0, 2))
    for i in range(start_num, end_num):
        # Construct the full file paths
        prev_image_path =  f"{base_name}{i:02d}_Z_fwd.png"# i的两位数字表示
        curr_image_path =  f"{base_name}{i + 1:02d}_Z_fwd.png"
        save_image_path =  f"./{save_dir}"
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)
        save_image_name =  f"./{save_dir}/{base_name}{i + 1:02d}_Z_fwd_tracked.png"

        # Load the images
        prev_image = cv2.imread(prev_image_path)
        curr_image = cv2.imread(curr_image_path)

        # Check if images were loaded
        if prev_image is None or curr_image is None:
            print(f"Failed to load images for iteration {i}.")
            continue

        # 计算漂移
        if i==start_num:
            # Calculate the shift of the region
            drift_region = drift_method.calculate_shift_with_region_tracking(prev_image, curr_image, region_start)
            region_new = region_start
        else:
            drift_region = drift_method.calculate_shift_with_region_tracking(prev_image, curr_image, region_new)
        # calculate_shift_with_region_tracking calculate_shift_with_closest_match
        print(f"drift_region:{drift_region}\n")
        result = np.vstack((result, drift_region))

        # Draw the region on the first image
        image_with_region_drawn = drift_method.draw_region_on_image(prev_image, region_new)
        # Calculate the new location of the region in the second image
        region_new_x = region_new[0] + drift_region[0]
        region_new_y = region_new[1] + drift_region[1]
        region_new = [region_new_x, region_new_y, region_size_x, region_size_y]
        # Draw the region on the second image
        image_with_region_new_drawn = drift_method.draw_region_on_image(curr_image, region_new)
        # data_pic.show_images(image_with_region_drawn, image_with_region_new_drawn)

        # 保存绘制特征点的图片
        if i==start_num:
            # cv2.imwrite(save_image_name, image_with_region_drawn)
            cv2.imwrite(f"./{save_dir}/{base_name}{i:02d}_Z_fwd_tracked.png", image_with_region_drawn, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # cv2.imwrite(save_image_name, image_with_region_new_drawn)
        cv2.imwrite(save_image_name, image_with_region_new_drawn, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        print(f"Processed and saved: {save_image_name}")
    return result

def process_images_regions( start_num, end_num, save_dir,base_name):
    # 创建一个空的 1x2 的 NumPy 数组
    result = np.empty((0, 2))
    for i in range(start_num, end_num):
        # Construct the full file paths
        prev_image_path =  f"{base_name}{i:02d}_Z_fwd.png"# i的两位数字表示
        curr_image_path =  f"{base_name}{i + 1:02d}_Z_fwd.png"
        save_image_path =  f"./{save_dir}"
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)
        save_image_name =  f"./{save_dir}/{base_name}{i + 1:02d}_Z_fwd_tracked.png"

        # Load the images
        prev_image = cv2.imread(prev_image_path)
        curr_image = cv2.imread(curr_image_path)


        # Check if images were loaded
        if prev_image is None or curr_image is None:
            print(f"Failed to load images for iteration {i}.")
            continue

        # 计算漂移并绘制特征点
        drift_regions = [0, 0]
        if i==start_num:
            # Calculate the shift of the region
            drift_regions[0], drift_regions[1], image1_with_features, image2_with_features = \
                drift_method.calculate_shift_and_mark_spots_corrected(prev_image, curr_image)
        else:
            drift_regions[0], drift_regions[1], image1_with_features, image2_with_features = \
                drift_method.calculate_shift_and_mark_spots_corrected(prev_image, curr_image)
        print(f"drift_regions:{drift_regions}\n")
        # data_pic.show_images(image1_with_features, image2_with_features)
        result = np.vstack((result, drift_regions))

        if i==start_num:
            # cv2.imwrite(save_image_name, image_with_region_drawn)
            cv2.imwrite(f"./{save_dir}/{base_name}{i:02d}_Z_fwd_tracked.png", image1_with_features, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # cv2.imwrite(save_image_name, image_with_region_new_drawn)
        cv2.imwrite(save_image_name, image2_with_features, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # dpi = 100  # Set the desired DPI
        # fig_size_x = z_processed.shape[0] / dpi  # Calculate the figure size for allocated pixels
        # fig_size_y = z_processed.shape[1] / dpi  # Calculate the figure size for allocated pixels
        # plt.figure(figsize=(fig_size_x, fig_size_y), dpi=dpi)  # Set figure size
        # if (save_fig):
        #     plt.savefig(dir_figs + '/' + self.fname[:-4] + '_' + i.name + '.png',
        #                 dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0)
        #     # 官方解释：bbox_inches控制要保存的图像的边界框(boundingbox)。
        #     # 默认情况下，它的值为'tight'，这意味着Matplotlib会尽量剪裁图像周围的空白区域，
        #     # 使图像尽可能地紧凑，以包含所有图形元素。
        #     # 使用方式：plt.savefig('output.png', bbox_inches='tight')
        #     # 官方解释：pad_inches控制图像保存时在边界框周围添加的填充。
        #     # 默认情况下，它的值为0.1，表示在边界框周围添加0.1英寸的填充。
        #     # 使用方式：plt.savefig('output.png', pad_inches=0)
        # if (show):
        #     plt.show()
        print(f"Processed and saved: {save_image_name}")
    return result




def find_common_and_extreme_parts(path):
    """
    Find common parts, the extreme values of non-common parts, and their bit width in the filenames within a given directory.

    :param path: Path to the directory
    :return: Tuple containing common prefix, common suffix, minimum and maximum of non-common parts, and bit width
    """
    if not os.path.isdir(path):
        return "Provided path is not a directory."

    # Extract all filenames in the directory
    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if len(filenames) < 2:
        return "Need at least two files to find common parts and extremes."

    # Function to find common prefix and suffix
    def common_prefix_suffix(str1, str2):
        # Common prefix
        prefix = os.path.commonprefix([str1, str2])

        # Common suffix
        min_length = min(len(str1), len(str2))
        suffix = ''
        for i in range(1, min_length + 1):
            if str1[-i:] == str2[-i:]:
                suffix = str1[-i:]
            else:
                break

        return prefix, suffix

    # Initialize common prefix and suffix with the first two filenames
    common_prefix, common_suffix = common_prefix_suffix(filenames[0], filenames[1])

    # Find common prefix and suffix among all files
    for filename in filenames[2:]:
        print("\033[32m", common_prefix, common_suffix,filename, "\033[0m")
        new_prefix, new_suffix = common_prefix_suffix(common_prefix+'\n'+common_suffix, filename)
        common_prefix = new_prefix
        common_suffix = new_suffix

    # Remove common parts to find the non-common parts
    non_common_parts = [filename.replace(common_prefix, '', 1).replace(common_suffix, '', 1) for filename in filenames]

    # Convert non-common parts to integers if possible and calculate bit width
    try:
        non_common_numbers = [int(part) for part in non_common_parts]
        min_non_common = min(non_common_numbers)
        max_non_common = max(non_common_numbers)
        bit_width = max_non_common.bit_length()
    except ValueError:
        # If conversion to integer fails, return the non-numeric values and bit width as 0
        min_non_common = min(non_common_parts)
        max_non_common = max(non_common_parts)
        bit_width = 0

    return common_prefix, common_suffix, min_non_common, max_non_common, bit_width


    # Example usage (You need to specify an actual path to test)
    # common_extremes = find_common_and_extreme_parts("/path/to/directory")
    # print(common_extremes)

def pad_minimum_value(min_value, bit_width):
    """
    Pad the minimum value with leading zeros based on the bit width.

    :param min_value: The minimum value to be padded
    :param bit_width: The bit width to determine the padding length
    :return: Padded minimum value as a string
    """
    # Calculate the number of digits required for the maximum value that can be represented by the bit width
    max_value = 2 ** bit_width - 1
    num_digits = len(str(max_value))

    # Pad the minimum value with leading zeros to match the number of digits
    padded_min_value = str(min_value).zfill(num_digits)

    return padded_min_value



if __name__=="__main__":
    # 图像生成及导入
    use_real=1
    show=0
    if use_real:
        path_base = r'C:\Users\14170\Desktop\python\openSPM\sxm_process\fig'
        path_sub= r"\0406晚drift\au111-0406034"
        # r"\0408晚drift\au111-230408049"
        # r"\0406晚drift\au111-0406001"
        # r"\0406晚drift\au111-0406027"
        # r"\0406晚drift\au111-0406034"
        # r"\202306 drift"
        # r"\202303 drift"
        common_prefix,common_suffix,name_min,name_max,bit_width = find_common_and_extreme_parts(path_base+path_sub)
        print("\033[32m",common_prefix,common_suffix,name_min,name_max,bit_width,"\033[0m")
        # 'Ag110-230213-drift0'
        # 'au111-2304080'
        os.chdir(path_base+path_sub)
        # Example usage
        padded_min_value = pad_minimum_value(name_min, bit_width)

        image1 = cv2.imread(common_prefix+ f'{pad_minimum_value(name_min, bit_width)}'+common_suffix)
        image2 = cv2.imread(common_prefix+ f'{pad_minimum_value(name_min+1, bit_width)}'+common_suffix)
        true_drift='please measure'

    else:
        true_drift = center_dx, center_dy = 100, 0  # Example center shift values
        image1, image2 = data_pic.create_images_with_centered_circles(center_dx, center_dy)
    region_x_realtives={r"\0406晚drift\au111-0406001":110,r"\0406晚drift\au111-0406027":75    ,r"\0406晚drift\au111-0406034":50,r'\202303 drift':240   ,r'\202306 drift':240   ,r"\0408晚drift\au111-230408049":108}
    region_y_realtives={r"\0406晚drift\au111-0406001":55 ,r"\0406晚drift\au111-0406027":375   ,r"\0406晚drift\au111-0406034":50,r'\202303 drift':340   ,r'\202306 drift':24    ,r"\0408晚drift\au111-230408049":165}
    region_size_x_dic ={r"\0406晚drift\au111-0406001":20 ,r"\0406晚drift\au111-0406027":20    ,r"\0406晚drift\au111-0406034":50,r'\202303 drift':15    ,r'\202306 drift':43    ,r"\0408晚drift\au111-230408049":20}
    region_size_y_dic ={r"\0406晚drift\au111-0406001":20 ,r"\0406晚drift\au111-0406027":20    ,r"\0406晚drift\au111-0406034":50,r'\202303 drift':15    ,r'\202306 drift':48    ,r"\0408晚drift\au111-230408049":20}
######################################################################################################
    # Define the region to track
    region_x      = region_x_realtives[path_sub]# x中心点为image1.shape[0] // 2,正方向向右
    region_y      = region_y_realtives[path_sub]# y中心点为image1.shape[1] // 2,正方向向下
    region_size_x = region_size_x_dic[path_sub]
    region_size_y = region_size_y_dic[path_sub]
    region = [region_x, region_y, region_size_x, region_size_y]
    print('region:',region)
    # Calculate the shift of the region
    drift_region=drift_method.calculate_shift_with_region_tracking(image1, image2, region)
    # Draw the region on the first image
    image_with_region_drawn = drift_method.draw_region_on_image(image1, region)
    # Calculate the new location of the region in the second image
    region_new_x = region_x + drift_region[0]
    region_new_y = region_y + drift_region[1]
    region_new = [region_new_x, region_new_y, region_size_x, region_size_y]
    # Draw the region on the second image
    image_with_region_new_drawn = drift_method.draw_region_on_image(image2, region_new)
    if show:
        data_pic.show_images(image_with_region_drawn, image_with_region_new_drawn)

    # Use this function with the appropriate parameters
    # base_path is the directory where your images are located
    # start_num is the starting image number (1 for au111-230408001_Z_fwd_.png)
    # end_num is the ending image number (49 for au111-230408050_Z_fwd_.png, since the loop ends at end_num - 1)
    drift_region_arr=process_images( name_min, name_max , "tracked",region,common_prefix)
    # 分别提取 x 和 y 数据列
    x = drift_region_arr[:, 0]
    y = drift_region_arr[:, 1]

    ##################################################################################################
    fig=data_pic.pic_movement(x,y)
    pic_name_and_path=path_base + path_sub + r'\tracked\distribution.svg'
    plt.savefig(pic_name_and_path, format='svg')
    # 将图片存储到论文图片路径下
    save_pic_type='svg'
    path_pic = r'C:\Users\14170\Desktop\SPM\毕业事宜及论文\论文图片\drift-tracked-statistic\\'
    pic_name = 'distribution'+path_sub.replace("\\", "-")+'region'
    pic_name_and_path = path_pic + pic_name  # + '.png'#'.svg'
    plt.savefig(pic_name_and_path + '.' + save_pic_type, format=save_pic_type)
    if show:
        plt.show()



    ##################################################################################################
    # 计算漂移并绘制特征点
    drift_regions=[0,0]
    drift_regions[0], drift_regions[1], image1_with_features, image2_with_features = \
        drift_method.calculate_shift_and_mark_spots_corrected(image1, image2)
    # drift_method.calculate_shift_and_mark_spots(image1, image2)
    # 显示图像
    if show:
        cv2.imshow("Image 1 with Features", image1_with_features)
        cv2.imshow("Image 2 with Features", image2_with_features)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    drift_regions_arr=process_images_regions( name_min, name_max , "tracked_regions",common_prefix)
    # 分别提取 x 和 y 数据列
    x = drift_regions_arr[:, 0]
    y = drift_regions_arr[:, 1]
    # 绘制图像
    # plt.figure(figsize=(8, 6))
    # plt.plot(x, y, marker='o', linestyle='-')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('X and Y')
    # plt.grid(True)
    # plt.show()
    fig=data_pic.pic_movement(x,y)
    pic_name_and_path=path_base + path_sub + r'\tracked_regions\distribution.svg'
    plt.savefig(pic_name_and_path,format='svg')
    # 将图片存储到论文图片路径下
    save_pic_type='svg'
    path_pic = r'C:\Users\14170\Desktop\SPM\毕业事宜及论文\论文图片\drift-tracked-statistic\\'
    pic_name = 'distribution'+path_sub.replace("\\", "-")+'regions'
    pic_name_and_path = path_pic + pic_name  # + '.png'#'.svg'
    plt.savefig(pic_name_and_path + '.' + save_pic_type, format=save_pic_type)
    if show:
        plt.show()



###################################################################################
    calculated_drift = drift_method.calculate_drift(image1, image2)
    box_drift = drift_method.calculate_box_drift(image1, image2)
    drift_orb = drift_method.calculate_image_orb(image1, image2)
    # 显示结果
    print(f"Drift:\ntrue_drift:{true_drift}\n"
          f"calculated_drift:{calculated_drift}\n"
          f"box_drift:{box_drift}\n"
          f"drift_orb:{drift_orb}\n"
          f"drift_region:{drift_region}\n"
          f"drift_regions:{drift_regions}\n")






