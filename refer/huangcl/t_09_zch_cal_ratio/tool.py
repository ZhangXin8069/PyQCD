import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pdb # 用于调试代码 pdb.set_trace()
from scipy import stats
from scipy.optimize import minimize_scalar


def visualize_matrix(matrix, name = 'test'):
    """
    Visualize a 2D matrix using color mapping.
    For complex matrices, create two separate plots for real and imaginary parts.
    Automatically adjusts ticks with equal spacing (4-10 ticks) including first/last.
    
    Parameters:
        matrix (numpy.ndarray): 2D matrix (can be real or complex)
        name (str): Base name for saving the images
    """
    # Check if input is a 2D matrix
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix")

    # Ensure the output directory exists
    os.makedirs('picture', exist_ok=True)

    # Get matrix dimensions
    rows, cols = matrix.shape
    
    def calculate_ticks(length):
        """Calculate optimal tick positions with equal spacing (4-10 ticks)"""
        if length <= 1:
            return [0]
        
        # Determine optimal number of ticks (4-10)
        n_ticks = min(10, max(4, length))
        
        # Calculate step size that gives equal spacing including last index
        step = (length - 1) / (n_ticks - 1)
        
        # Generate ticks with equal spacing
        ticks = [round(i * step) for i in range(n_ticks)]
        
        # Ensure first and last ticks are exactly 0 and length-1
        ticks[0] = 0
        ticks[-1] = length - 1
        
        # Remove duplicates that may occur due to rounding
        ticks = sorted(list(set(ticks)))
        
        return ticks
    
    x_ticks = calculate_ticks(cols)
    y_ticks = calculate_ticks(rows)

    if np.iscomplexobj(matrix):
        # Plot real part
        plt.figure()
        plt.imshow(matrix.real, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xticks(x_ticks, fontsize=10)
        plt.yticks(y_ticks, fontsize=10)
        plt.title(f"Real Part of {name}")
        plt.savefig(f'picture/{name}_real.pdf', format='pdf', bbox_inches='tight')
        plt.close()
        
        # Plot imaginary part
        plt.figure()
        plt.imshow(matrix.imag, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xticks(x_ticks, fontsize=10)
        plt.yticks(y_ticks, fontsize=10)
        plt.title(f"Imaginary Part of {name}")
        plt.savefig(f'picture/{name}_imag.pdf', format='pdf', bbox_inches='tight')
        plt.close()
        
    else:
        # Real matrix case
        plt.figure()
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xticks(x_ticks, fontsize=10)
        plt.yticks(y_ticks, fontsize=10)
        plt.title(f"Matrix {name}")
        plt.savefig(f'picture/{name}.pdf', format='pdf', bbox_inches='tight')
        plt.close()

def visualize_matrix_num(matrix, name = 'test'):
    """
    Visualize a 2D matrix using color mapping with values displayed in each cell.
    For complex matrices, create two separate plots for real and imaginary parts.
    Automatically adjusts ticks with equal spacing (4-10 ticks) including first/last.
    
    Parameters:
        matrix (numpy.ndarray): 2D matrix (can be real or complex)
        name (str): Base name for saving the images
    """
    # Check if input is a 2D matrix
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix")

    # Ensure the output directory exists
    os.makedirs('picture', exist_ok=True)

    # Get matrix dimensions
    rows, cols = matrix.shape
    
    def calculate_ticks(length):
        """Calculate optimal tick positions with equal spacing (4-10 ticks)"""
        if length <= 1:
            return [0]
        
        # Determine optimal number of ticks (4-10)
        n_ticks = min(10, max(4, length))
        
        # Calculate step size that gives equal spacing including last index
        step = (length - 1) / (n_ticks - 1)
        
        # Generate ticks with equal spacing
        ticks = [round(i * step) for i in range(n_ticks)]
        
        # Ensure first and last ticks are exactly 0 and length-1
        ticks[0] = 0
        ticks[-1] = length - 1
        
        # Remove duplicates that may occur due to rounding
        ticks = sorted(list(set(ticks)))
        
        return ticks
    
    x_ticks = calculate_ticks(cols)
    y_ticks = calculate_ticks(rows)

    def plot_matrix(data, title, filename):
        """Helper function to plot a single matrix with values"""
        plt.figure(figsize=(max(cols/2, 6), max(rows/2, 6)))  # Adjust figure size based on matrix size
        
        # Create the heatmap
        im = plt.imshow(data, cmap='viridis', interpolation='nearest')
        plt.colorbar(im)
        
        # Add text annotations
        for i in range(rows):
            for j in range(cols):
                # Format the number to 3 significant digits using scientific notation
                text = f"{data[i, j]:.1g}"
                # Choose text color based on cell brightness
                color = 'white' if data[i, j] < np.mean(data) else 'black'
                # Place text in the center of the cell
                plt.text(j, i, text,
                        ha="center", va="center",
                        color=color, 
                        fontsize=min(12, 120/max(rows, cols),  # 调整基础字体大小
                                    # 根据单元格大小进一步调整
                                    min(150/cols, 150/rows)),  # 限制最大字体大小
                        fontweight='bold')  #
        
        plt.xticks(x_ticks, fontsize=10)
        plt.yticks(y_ticks, fontsize=10)
        plt.title(title)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()

    if np.iscomplexobj(matrix):
        # Plot real part
        plot_matrix(matrix.real, f"Real Part of {name}", f'picture/{name}_real.pdf')
        
        # Plot imaginary part
        plot_matrix(matrix.imag, f"Imaginary Part of {name}", f'picture/{name}_imag.pdf')
        
    else:
        # Real matrix case
        plot_matrix(matrix, f"Matrix {name}", f'picture/{name}.pdf')

def hist_draw(data_array, bins=80, filename='distribution.pdf'):
    """
    绘制一维数组的分布直方图并保存图片
    
    参数:
    data_array : 一维数组或列表，输入数据
    bins : int或序列，直方图的bin数量或边界，默认为10
    filename : str, 保存的文件名，默认为'distribution.png'
    """
    plt.figure(figsize=(8, 6))
    

    data_array_cut = data_array

    #print(data_array_cut)

    plt.hist(data_array_cut, bins=bins, edgecolor='black', alpha=0.7)

    

    # 计算平均值
    mean_val = np.mean(data_array_cut)
    mean_std = np.std(data_array_cut)
    
    # 在平均值处添加垂直线
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {mean_val:.2f}')
    
    plt.text(mean_val, plt.ylim()[1] * 0.9, f'All samples mean = {mean_val}', color='red', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(mean_val, plt.ylim()[1] * 0.8, f'All samples std = {mean_std}',  color='red', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.title('Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    #plt.title(r'L32x96, Pz = 0, a=0.0775, $\Delta t= 2$')
    #plt.title(r'L32x64, Pz = 0, a=0.0897, $\Delta t= 2$')
    #plt.title(r'L24x72, Pz = 0, a=0.105,  $\Delta t= 2$')
    #plt.xlabel('2pt')
    #plt.ylabel('Frequency')
    
    # 添加网格线
    plt.grid(axis='y', alpha=0.5)
    
    # 保存图片
    plt.savefig(filename)
   
def jackknife(data, axis=0):
    """
    计算N维数组中每个元素在排除自身后，沿指定轴的均值（Jackknife估计）
    
    数学表达式：
    对于输入数组data，输出数组中每个元素的值 = (沿axis轴的总和 - 当前元素) / (沿axis轴的元素数 - 1)
    
    参数：
        data : numpy.ndarray
            输入的多维数组
        axis : int
            计算均值的轴向（如0=行方向，1=列方向）
            
    返回：
        numpy.ndarray
            与输入数组形状相同的Jackknife估计结果
    """
    # 沿指定轴求和，并保持维度（便于广播）
    total = np.sum(data, axis=axis, keepdims=True)
    
    # 获取指定轴的长度（即参与求和的元素数量）
    count = data.shape[axis]
    
    # 计算排除当前元素后的均值：
    # 1. (total - data) -> 从总和中减去当前元素（广播机制自动处理维度）
    # 2. / (count - 1)  -> 除以剩余元素数量
    return (total - data) / (count - 1)

def bootstrap(data, n_resamples= 3000 , axis=0, random_seed=1227):
    """
    计算N维数组沿指定轴的Bootstrap重采样均值分布
    
    数学表达式：
    对于输入数组data，执行B次重采样（B=n_resamples）：
    1. 每次从data的axis轴随机抽取n个样本（n=data.shape[axis]），允许重复
    2. 计算每次重采样后的沿axis轴的均值
    最终返回所有重采样均值的集合
    
    参数：
        data : numpy.ndarray
            输入的多维数组，形状为(..., n_samples, ...)
        n_resamples : int
            重采样次数（默认3000次）
        axis : int
            计算均值的轴向（如0=行方向，1=列方向）
        random_seed : int
            随机种子（保证结果可复现）
            
    返回：
        numpy.ndarray
            形状为(n_resamples, ...)的数组，包含所有重采样均值
    """
    # 设置随机种子（如果提供）
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 获取原始数据沿指定轴的样本数量
    n_samples = data.shape[axis]
    
    # 生成随机索引矩阵（形状：n_resamples × n_samples）
    indices = np.random.randint(0, n_samples, size=(n_resamples, n_samples))
    
    # 使用高级索引获取所有重采样数据（避免循环）
    # 说明：np.take允许沿指定轴索引，比直接索引更通用
    resampled_data = np.take(data, indices, axis=axis)
    
    # 计算每次重采样的均值（沿样本轴）
    return np.mean(resampled_data, axis=axis+1)  # axis+1因为新增了n_resamples维度

def bootstrap_new(data, n_resamples=3000, axis=0, random_seed=1000, chunk_size=100):
    """
    分块Bootstrap重采样，避免内存爆炸。
    
    参数：
        data : numpy.ndarray
            输入数据（如形状 (777, 30, 26, 31)）
        chunk_size : int
            每块的重采样次数（默认100）
    """
    np.random.seed(random_seed)
    n_samples = data.shape[axis]
    result = []
    
    for i in range(0, n_resamples, chunk_size):
        # 1. 生成当前块的随机索引
        current_chunk_size = min(chunk_size, n_resamples - i)
        indices = np.random.randint(0, n_samples, size=(current_chunk_size, n_samples))
        
        # 2. 获取当前块的重采样数据
        chunk_data = np.take(data, indices, axis=axis)  # 形状 (chunk_size, ...)
        
        # 3. 计算当前块的均值
        chunk_means = np.mean(chunk_data, axis=axis + 1)
        result.append(chunk_means)
    
    # 合并所有块的结果
    return np.concatenate(result, axis=0)

def boot(a,Nconf_new):   #for a:the last dimension is configuration
    print("boot")
    nsample=a.shape[0]
    #nsample=2
    #a_T=np.swapaxes(a, 0, -1)
    new_shape = (Nconf_new,) + a.shape[1:]
    y=np.zeros(new_shape)
    random.seed(1227)
    for N in range(Nconf_new):
        y_N=np.zeros((nsample,) + a.shape[1:])
        for n in range(nsample):
            y_N[n]=a[random.randint(0,nsample-1)]
        y_N_aver=np.average(y_N,axis=0)    #a.shape[0]
        y[N]=y_N_aver
    #y=np.swapaxes(y_T, 0, -1)
    #print()
    return y    #a.shape[0],Nconf_new

def covariance_matrix_inv(data, Resam_):
    """
    计算协方差矩阵（样本间协方差，输出 a×a 矩阵）的逆
    
    参数:
        data : numpy.ndarray, 形状 (a, b)
            - a: 数据个数（样本数）
            - b: 每个数据的样本数（变量数）
    
    返回:
        cov_matrix : numpy.ndarray, 形状 (a, a)
            - 样本间的协方差矩阵
    """
    # 中心化数据（每行减去均值）
    n_len = data.shape[1]
    centered_data = data - np.mean(data, axis=1, keepdims=True)
    
    # 计算协方差矩阵 (使用 (X - μ)(X - μ)^T / (b - 1))
    cov_matrix = np.dot(centered_data, centered_data.T) / (data.shape[1] - 1)
    #return cov_matrix
    if Resam_ == 'jack':
        return np.linalg.inv(cov_matrix * (n_len - 1 ) )
    else:
        return np.linalg.inv(cov_matrix)
    
def covariance_matrix(data, Resam_):
    """
    计算协方差矩阵（样本间协方差，输出 a×a 矩阵）的逆
    
    参数:
        data : numpy.ndarray, 形状 (a, b)
            - a: 数据个数（样本数）
            - b: 每个数据的样本数（变量数）
    
    返回:
        cov_matrix : numpy.ndarray, 形状 (a, a)
            - 样本间的协方差矩阵
    """
    # 中心化数据（每行减去均值）
    n_len = data.shape[1]
    centered_data = data - np.mean(data, axis=1, keepdims=True)
    
    # 计算协方差矩阵 (使用 (X - μ)(X - μ)^T / (b - 1))
    cov_matrix = np.dot(centered_data, centered_data.T) / (data.shape[1] - 1)
    #return cov_matrix
    if Resam_ == 'jack':
        return cov_matrix * (n_len - 1 ) 
    else:
        return cov_matrix

def replace_small_singular_values(matrix, min_singular_value=1e-5):
    """
    对矩阵进行SVD分解，将小于最小奇异值的奇异值替换为最小奇异值
    
    Parameters:
    matrix: 输入矩阵
    min_singular_value: 最小奇异值阈值
    
    Returns:
    modified_matrix: 奇异值修改后重建的矩阵
    original_s: 原始的奇异值
    modified_s: 修改后的奇异值
    """

    # 转换为numpy数组
    A = np.array(matrix, dtype=float)
    
    # 进行SVD分解
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    
    # 记录小于阈值的奇异值个数
    small_count = np.sum(s < min_singular_value)
   
    # 修改奇异值
    s_modified = s.copy()
    s_modified[s_modified < min_singular_value] = min_singular_value
    
   
    # 重建矩阵
    modified_matrix = U @ np.diag(s_modified) @ Vt
    visualize_matrix_num(np.diag(s_modified), name = 'test')
    
    return modified_matrix

def symmetric_process_optimized(all_data):
    """
    优化版的对称化处理
    """
    data = all_data['data']
    samples = all_data['samples']
    
    data_new = data.copy()
    samples_new = samples.copy()
    
    # 使用字典快速查找对称位置
    position_dict = {}
    for i, (z, t_sep, ti_sep, mean, std) in enumerate(data):
        key = (z, t_sep)
        if key not in position_dict:
            position_dict[key] = {}
        position_dict[key][ti_sep] = i
    
    # 处理每个 (z, t_sep) 组合
    for (z, t_sep), ti_dict in position_dict.items():
        for ti_sep, idx in ti_dict.items():
            symmetric_ti_sep = t_sep - ti_sep
            
            # 如果对称位置存在且不同
            if symmetric_ti_sep != ti_sep and symmetric_ti_sep in ti_dict:
                symmetric_idx = ti_dict[symmetric_ti_sep]
                
                # 计算平均值
                avg_samples = (samples[idx] + samples[symmetric_idx]) / 2.0
                
                # 更新样本数据
                samples_new[idx] = avg_samples
                samples_new[symmetric_idx] = avg_samples
                
                # 更新统计信息
                new_mean = np.mean(avg_samples)
                new_std = np.std(avg_samples)
                
                data_new[idx][3] = new_mean
                data_new[idx][4] = new_std
                data_new[symmetric_idx][3] = new_mean
                data_new[symmetric_idx][4] = new_std
    
    return {'data': data_new, 'samples': samples_new}

def symmetric_process_fit(all_data):
    """
    优化版的对称化处理，对称化后只保留一半的数据点
    """
    data = all_data['data']
    samples = all_data['samples']
    
    data_new = []
    samples_new = []
    
    # 使用字典快速查找对称位置
    position_dict = {}
    for i, (z, t_sep, ti_sep, mean, std) in enumerate(data):
        key = (z, t_sep)
        if key not in position_dict:
            position_dict[key] = {}
        position_dict[key][ti_sep] = i
    
    # 记录已经处理过的位置，避免重复添加
    processed_pairs = set()
    
    # 处理每个 (z, t_sep) 组合
    for (z, t_sep), ti_dict in position_dict.items():
        # 计算中间点，用于判断保留哪一半
        mid_point = t_sep / 2.0
        
        for ti_sep, idx in ti_dict.items():
            symmetric_ti_sep = t_sep - ti_sep
            
            # 如果这个位置已经被处理过，跳过
            if (z, t_sep, ti_sep) in processed_pairs:
                continue
            
            # 如果对称位置存在且不同
            if symmetric_ti_sep != ti_sep and symmetric_ti_sep in ti_dict:
                symmetric_idx = ti_dict[symmetric_ti_sep]
                
                # 计算平均值
                avg_samples = (samples[idx] + samples[symmetric_idx]) / 2.0
                
                # 只保留 ti_sep <= mid_point 的数据点
                # 这样当 t_sep=9 时，mid_point=4.5，保留 ti_sep=0,1,2,3,4
                if ti_sep <= mid_point:
                    # 更新统计信息
                    new_mean = np.mean(avg_samples)
                    new_std = np.std(avg_samples)
                    
                    # 添加到新数据中
                    data_new.append([z, t_sep, ti_sep, new_mean, new_std])
                    samples_new.append(avg_samples)
                
                # 标记这两个位置都已处理
                processed_pairs.add((z, t_sep, ti_sep))
                processed_pairs.add((z, t_sep, symmetric_ti_sep))
            else:
                # 对称位置不存在或者是自身，直接保留
                # 对于 ti_sep = mid_point 的情况（当 t_sep 为偶数时）
                data_new.append(data[idx])
                samples_new.append(samples[idx])
                processed_pairs.add((z, t_sep, ti_sep))
    
    data_new    = np.array(data_new)
    samples_new = np.array(samples_new)
    return {'data': data_new, 'samples': samples_new}

def plot_scatter_with_correlation(array, row_indices, file_name, par_fit = None):
    """
    Plot a scatter diagram of two rows from a 2D array and display the correlation coefficient
    
    Parameters:
    array: 2D array (can be list, numpy array, etc.)
    row_indices: list or tuple containing two row indices, e.g., [0, 1] or (1, 2)
                 Note: row indices start from 0, so row 1 is index 0, row 2 is index 1
    file_name: name of the file to save the plot
    
    Returns:
    correlation: calculated correlation coefficient
    """
    save_way = '/public/group/imp/zengch/LQCD/renorma/picture/correlation_picture/'
    save_path = f'{save_way}{file_name}'
    
    # Save the original array to txt file
    save_2d_array_to_txt(array, save_path)

    

    # Convert to numpy array
    arr = np.array(array)
    
    # Check input
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array")
    
    if len(row_indices) != 2:
        raise ValueError("Must provide two row indices")
    
    row1, row2 = row_indices
    
    # Extract two rows of data
    x_data = arr[row1, :]
    y_data = arr[row2, :]

    def fit_form(lambda_, par_put_):
        [l1_, a1_, lambda0_] = par_put_
        res = l1_ * lambda_ **(-a1_) * np.exp(-lambda_/lambda0_)
        #print(res)
        #pdb.set_trace()
        return res


    def chi2_set_get():
        chi2_mean_set = []
        chi2_th_set=[]
        #c_inv = covariance_matrix_inv(array,'boot')

        
        
        c_test = covariance_matrix(array,'boot')
        c_test = replace_small_singular_values(c_test)
        c_inv  = np.linalg.inv(c_test)
        
       
        
        

        
        lenh, lenz = np.shape(array)


        r_mean = np.mean(array, axis = 1)

        r_th = r_mean
        if par_fit != None:
            var_ = par_fit[0]
            par_ = par_fit[1]
            r_th = fit_form(var_, par_) 

        print('r_mean:',r_mean)
        print('r_th',r_th)

        nr = len(r_mean)
        for i in range(lenz):
            r_i = array[:, i]
            r_m = r_i - r_mean
            r_t = r_i - r_th

            chi2_m = r_m.T @ c_inv @ r_m / nr
            chi2_t = r_t.T @ c_inv @ r_t / nr
            chi2_mean_set.append(chi2_m)
            chi2_th_set.append(chi2_t)
        return np.array(chi2_mean_set), np.array(chi2_th_set)

    chi2_mean_all, chi2_th_all = chi2_set_get()

    print('chi2 ='  ,chi2_mean_all)
    print('chi2 th=',chi2_th_all)

    print('chi2 mean=',np.mean(chi2_mean_all))
    print('chi2 mean th=',np.mean(chi2_th_all))
    
    # Calculate Pearson correlation coefficient
    correlation, p_value = stats.pearsonr(x_data, y_data)
    
    # Calculate center values (mean of each row)
    x_center = np.mean(x_data)
    y_center = np.mean(y_data)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot scatter points
    plt.scatter(x_data, y_data, alpha=0.6, edgecolors='black', linewidth=0.5, label='Data points')
    
    # Add trend line
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    plt.plot(x_data, p(x_data), "r--", alpha=0.8, label=f'Trend line (slope: {z[0]:.4f})')
    
    # Mark the center point
    plt.scatter(x_center, y_center, color='red', s=200, marker='*', 
                edgecolors='black', linewidth=1.5, zorder=5, label='Center point')
    
    # Add dashed lines from center to axes
    plt.axhline(y=y_center, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    plt.axvline(x=x_center, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Set title and labels
    plt.title(f'Row {row1+1} vs Row {row2+1}', fontsize=14)
    plt.xlabel(f'Row {row1+1} Data', fontsize=12)
    plt.ylabel(f'Row {row2+1} Data', fontsize=12)
    
    # Display correlation information on the plot
    textstr = f'Correlation r = {correlation:.4f}\n'
    textstr += f'r² = {correlation**2:.4f}\n'
    textstr += f'p-value = {p_value:.4e}\n'
    textstr += f'Center: ({x_center:.4f}, {y_center:.4f})'
    
    # Add text box on the plot - moved to top right corner
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')  # Changed to fixed position
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return correlation

def plot_points_with_center(points, output_pdf="scatter_center.pdf"):
    """
    Draw a scatter plot of points and mark the mean center, then save as PDF.
    Additionally, find and mark two points on the line x=y:
        - Point minimizing average Euclidean distance to all data points.
        - Point minimizing average Mahalanobis distance to all data points.
    Also display the average distances at the optimum points.

    Parameters:
        points: numpy.ndarray, shape (2, N) or (N, 2). If more than 2 rows, first two rows are used as (x, y).
        output_pdf: str, output PDF filename (default "scatter_center.pdf").
    """
    # ----- 确保 points 是 (2, N) 格式 -----
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    if points.shape[0] != 2:
        # 如果行数不是2，尝试转置为 (2, N)
        if points.shape[1] == 2:
            points = points.T
        else:
            # 若形状既不是 (2,N) 也不是 (N,2)，则强行取前两行作为坐标
            points = points[:2, :]

    # 提取横纵坐标（现在 points 一定是 (2, N)）
    x = points[0, :]
    y = points[1, :]

    # 计算均值中心
    center_x = np.mean(x)
    center_y = np.mean(y)

    # ---------- 1. 在直线 x=y 上找最小化平均欧式距离的点 ----------
    def avg_euclidean_dist(t):
        return np.mean(np.sqrt((x - t)**2 + (y - t)**2))

    t_min = min(x.min(), y.min()) - 1.0
    t_max = max(x.max(), y.max()) + 1.0
    res_euclidean = minimize_scalar(avg_euclidean_dist, bounds=(t_min, t_max), method='bounded')
    t_euclidean = res_euclidean.x
    point_euclidean = (t_euclidean, t_euclidean)
    avg_euclidean_val = avg_euclidean_dist(t_euclidean)

    # ---------- 2. 在直线 x=y 上找最小化平均马氏距离的点 ----------
    data_2d = np.vstack((x, y))           # (2, N)
    cov = np.cov(data_2d, bias=False)      # (2, 2)
    if np.linalg.cond(cov) > 1e12:
        cov += np.eye(2) * 1e-8
    inv_cov = np.linalg.inv(cov)

    def avg_mahalanobis_dist(t):
        delta = np.vstack((x - t, y - t))
        sq_dist = np.sum((inv_cov @ delta) * delta, axis=0)
        sq_dist = np.maximum(sq_dist, 0.0)
        return np.mean(np.sqrt(sq_dist))

    res_mahalanobis = minimize_scalar(avg_mahalanobis_dist, bounds=(t_min, t_max), method='bounded')
    t_mahalanobis = res_mahalanobis.x
    point_mahalanobis = (t_mahalanobis, t_mahalanobis)
    avg_mahalanobis_val = avg_mahalanobis_dist(t_mahalanobis)

    print(f"Point on x=y minimizing average Euclidean distance: ({point_euclidean[0]:.4f}, {point_euclidean[1]:.4f})")
    print(f"Average Euclidean distance at that point: {avg_euclidean_val:.4f}")
    print(f"Point on x=y minimizing average Mahalanobis distance: ({point_mahalanobis[0]:.4f}, {point_mahalanobis[1]:.4f})")
    print(f"Average Mahalanobis distance at that point: {avg_mahalanobis_val:.4f}")

    # ---------- 绘图部分 ----------
    plt.figure(figsize=(8, 8))

    plt.scatter(x, y, c='blue', alpha=0.7, s=10, label='Data points')
    plt.scatter(center_x, center_y, c='red', marker='*', s=200,
                edgecolors='black', label=f'Center ({center_x:.2f}, {center_y:.2f})')
    plt.scatter(*point_euclidean, c='green', marker='^', s=120,
                edgecolors='black', label=f'Min Avg Eucl on x=y ({point_euclidean[0]:.2f}, {point_euclidean[1]:.2f})')
    plt.scatter(*point_mahalanobis, c='orange', marker='D', s=120,
                edgecolors='black', label=f'Min Avg Mahal on x=y ({point_mahalanobis[0]:.2f}, {point_mahalanobis[1]:.2f})')

    # 图例放置在左上角（固定位置）
    plt.legend(loc='upper left')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Scatter plot with center and optimal points on x=y')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 自动调整坐标轴范围
    x_margin = (x.max() - x.min()) * 0.1 if x.max() != x.min() else 1.0
    y_margin = (y.max() - y.min()) * 0.1 if y.max() != y.min() else 1.0
    plt.xlim(x.min() - x_margin, x.max() + x_margin)
    plt.ylim(y.min() - y_margin, y.max() + y_margin)

    ax = plt.gca()
    ax.set_aspect('equal')

    # 文本框放置在右上角（右对齐，避免与左上角图例重叠）
    info_text = (f"Average Euclidean distance (min point): {avg_euclidean_val:.4f}\n"
                 f"Average Mahalanobis distance (min point): {avg_mahalanobis_val:.4f}")
    plt.text(0.98, 0.98, info_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 保存 PDF
    plt.savefig(f'picture/{output_pdf}', format='pdf', bbox_inches='tight')
    print(f"Graph saved as: {output_pdf}")

    plt.close()

def save_2d_array_to_txt(array, filename):
    arr = np.array(array)
  
    np.savetxt(f'{filename}', arr.T, fmt='%s', delimiter='\t')
    print(f"数组已保存到{filename}")

def block_diagonal_mask(mat: np.ndarray, block_sizes: list) -> np.ndarray:
    """
    对角分块保留矩阵元素，其余置0
    :param mat: 输入二维方阵 (N, N)
    :param block_sizes: 对角分块尺寸列表，如 [3,2,2,4,3,3,3,3]
    :return: 仅保留对角分块、其余为0的同尺寸矩阵
    """
    # 校验输入是方阵
    n_row, n_col = mat.shape
    if n_row != n_col:
        raise ValueError("输入必须是二维方阵")
    total_dim = sum(block_sizes)
    if total_dim != n_row:
        raise ValueError(f"分块总尺寸 {total_dim} 与矩阵维度 {n_row} 不匹配")

    # 初始化全零输出矩阵
    out_mat = np.zeros_like(mat, dtype=mat.dtype)
    start = 0
    for blk_len in block_sizes:
        end = start + blk_len
        # 拷贝对角子块
        out_mat[start:end, start:end] = mat[start:end, start:end]
        start = end
    return out_mat


# 示例：创建一个随机矩阵并可视化
if __name__ == "__main__":
    # 创建一个 10x10 的随机矩阵
    #matrix = np.random.rand(7, 7)
    
    # 调用函数进行可视化
    #visualize_matrix(matrix, '1')

   
    
    mu = 0      # 中心值（均值）
    sigma = 1   # 标准差
    N = 20

    np.random.seed(39)

    a = np.random.normal(mu, sigma, N)
    a = np.array(a)

    print(a)
    print('mean:', np.mean(a))
    print('std:',  np.std(a))

    b = bootstrap_new(a, 20)

    print(b)
    print('boot_mean:', np.mean(b))
    print('boot_std:', np.std(b), np.std(b) * np.sqrt(N))

    
    c = jackknife(a)
    print(c)

    print('jack_mean:',np.mean(c))
    print('jack_std:',  np.std(c), np.std(c) * np.sqrt(N) * np.sqrt(N -1))

