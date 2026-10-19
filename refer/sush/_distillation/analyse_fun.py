import numpy as np
from scipy.optimize import fsolve
import fileinput
import sympy as sp
import gvar as gv
import lsqfit
from typing import List, Literal
import math
from opt_einsum import contract
import itertools
from scipy.linalg import det

def levi_civita_tensor(n:int = 3):
    """
    生成 n 维空间中的 Levi-Civita 张量
    
    参数:
        n (int): 张量的阶数（也是空间的维度）
        
    返回:
        np.ndarray: n 阶张量，形状为 (n, n, ..., n) [共 n 个维度]
    """
    # 创建 n×n×...×n 的零张量
    shape = tuple([n] * n)
    epsilon = np.zeros(shape, dtype = float)
    
    # 生成所有可能的索引排列
    for perm in itertools.permutations(range(n)):
        # 检查是否包含重复索引
        if len(set(perm)) == n:
            # 创建排列矩阵
            perm_matrix = np.eye(n)[list(perm)]
            
            # 计算行列式作为符号
            sign = det(perm_matrix)
            
            # 四舍五入避免浮点误差
            sign = np.round(sign).astype(int)
            
            # 赋值
            epsilon[perm] = sign
            
    return epsilon

def PFF_Mom_to_gamma_new(Mom, allow_t:bool = False):

	from opt_einsum import contract
	from itertools import combinations

	gamma_indx_list_matrix = [[[]]]
		

	if allow_t == False:
		lc_tensor = levi_civita_tensor(3)
		Mom_list = [x[::-1] for x in Mom]

	else:
		lc_tensor = levi_civita_tensor(4)
		Mom_list = [([1] + x)[::-1] for x in Mom]

	if Mom_list == [[0, 0, 0]]:
		gamma_indx_list_matrix = np.asarray([[[x, y] for x in range(1, 5) for y in range(1, 5)]])

	else:
		for _Mom in Mom_list:
			k = []
			k = [x_indx for x_indx, x in enumerate(_Mom) if x != 0]

			for l in np.asarray(list(combinations(k, lc_tensor.ndim - 2))):	  
				gamma_indx_list = [[]]
				if lc_tensor.ndim - 2 == 1:
					gamma_indx_matrix = lc_tensor[..., l[0]]
				
				elif lc_tensor.ndim - 2 == 2:
					gamma_indx_matrix = lc_tensor[..., l[0], l[1]]

				gamma_indx = np.argwhere(gamma_indx_matrix != 0) + 1
				for i in list(gamma_indx):
					i = [int(x) for x in i]
					gamma_indx_list += [i]

				gamma_indx_list_matrix += [gamma_indx_list[1:]]

		gamma_indx_list_matrix = np.asarray(gamma_indx_list_matrix[1:]).reshape(-1, len(list(combinations(k, lc_tensor.ndim - 2)))*2, 2)
			
	gamma_indx_list_all = np.asarray([[x for x in [1, 2, 3, 4] if x in gamma_indx_list_matrix[y]] for y in range(len(Mom))])

	return gamma_indx_list_matrix

def param_position(param:Literal['Mom', 'gamma', 'Nev', 'tsep', 'link_list', 'conf', 0, 1, 2, 3, 4, 5]):
    if param == 'Mom':
        return 0
    
    elif param == 'gamma':
        return 1
    
    elif param == 'Nev':
        return 2
    
    elif param == 'tsep':
        return 3
    
    elif param == 'link_list':
        return 4
    
    elif param == 'conf':
        return 5
    
    elif param == 0:
        return 'Mom'

    elif param == 1:
        return 'gamma'

    elif param == 2:
        return 'Nev'

    elif param == 3:
        return 'tsep'

    elif param == 4:
        return 'link_list'

    elif param == 5:
        return 'conf'


def adjust_duplicates_ordered(_list:list):
    # this func is to make the same value of _list become the value + n, n is the number of times the value appears.

    seen = set()
    mapping = {}  # 记录每个原始值最后分配到的值
    
    for i in range(len(_list)):
        num = _list[i]
        
        # 如果数字已存在，需要调整
        if num in seen:
            # 计算起始值：如果之前调整过，从最后分配值+1开始
            start = mapping.get(num, num) + 1
            # 找到第一个可用的值
            while start in seen:
                start += 1
            _list[i] = start
            mapping[num] = start
            seen.add(start)
        else:
            seen.add(num)
            # 记录原始值对应的初始值
            if num not in mapping:
                mapping[num] = num
    
    return _list

def create_data_info(
    file_3pt_path:str = '', 
    file_2pt_path:str = '', 

    conf:list = [],

    dtype:Literal['PDF', 'PFF'] = 'PDF',

    file_2pt_mu_nu_path:str = '', 
    file_bubble_path:str = '', 

    Nx:int = 24, 
    Nt:int = 72, 
    alttc:float = 0.10,

    Nev:list = [100], 
    tsep:list = [0], 
    link_list:list = [0], 
    Mom:list = [[0, 0, 0]],
    gamma:list = [[[]]],

    read_type:List[Literal['2pt', '3pt', '2pt_mu_nu', 'bubble', '2pt_all_t', '3pt_all_t']] = ['2pt'],
    meff_type:Literal['cosh', 'log'] = 'cosh',

    ):
    
    data_info_file = {}
    if '3pt_all_t' in read_type:
        data_info_file['file_3pt_all_t'] = f'{file_3pt_path}'
        
    if '2pt_all_t' in read_type:
        data_info_file['file_2pt_all_t'] = f'{file_2pt_path}'
        
    data_info_file['file_3pt'] = f'{file_3pt_path}'
    data_info_file['file_2pt'] = f'{file_2pt_path}'


    data_info_file['Nx'] = Nx
    data_info_file['Nt'] = Nt
    data_info_file['alttc'] = alttc

    data_info_file['Nev'] = Nev
    data_info_file['tsep'] = tsep
    data_info_file['link_list'] = link_list


    data_info_file['Mom'] = Mom

    data_info_file['conf'] = conf
    
    data_info_file['meff_type'] = meff_type
    data_info_file['read_type'] = read_type
    data_info_file['dtype'] = dtype

    data_info_file['gamma'] = gamma.copy()
    data_info_file['gamma_mu'] = gamma.copy()
    data_info_file['gamma_nu'] = gamma.copy()
    
    data_info_file['file_2pt_mu_nu'] = f'{file_2pt_mu_nu_path}'
    data_info_file['file_bubble'] = f'{file_bubble_path}'

    if dtype == 'PFF':
        # transport the param Mom to gamma use the way of PFF.
        gamma_ndarray = PFF_Mom_to_gamma_new(Mom, allow_t = False)
        
        data_info_file['gamma'] = gamma_ndarray.tolist()
        data_info_file['gamma_mu'] = gamma_ndarray[:, :, 1:2].tolist()
        data_info_file['gamma_nu'] = gamma_ndarray[:, :, 0:1].tolist()

    return data_info_file

def create_file_name(data_info, Ctype:Literal['2pt', '3pt', '2pt_mu_nu', 'bubble'] = '2pt', check_file:bool = True, **kwargs):
    if Ctype in data_info['read_type']:
        
        _file_init = data_info[f'file_{Ctype}']
        for pname, kname in kwargs.items():
            if 'Nx'  == kname or 'Nt' == kname:
                _file_init = _file_init.replace(f'<{pname}>', f'{data_info[kname]}')
        
        file_init = [_file_init]
        
        file_name = ['0']

        N = [1, 1, 1, 1, 1, 1]
        position = [5, 5, 5, 5, 5, 5]
        
        k = 1
        for pname, kname in kwargs.items():
            if 'Nx' != kname and 'Nt' != kname:
                if kname == 'gamma' and data_info['dtype'] == 'PFF':
                    raise ValueError('when dtype == PFF this fun don\'t support creat gamma list')

                position[param_position(kname)] -= k
                if kname == 'Mom':
                    for Mom_indx, Mom in enumerate(data_info[kname]):
                        _Mom = f'Px{Mom[2]}Py{Mom[1]}Pz{Mom[0]}'

                        for j in range(np.prod(N)):
                            file_name_Mom = file_init[j].replace(f'<{pname}>', _Mom)
                            
                            if data_info['dtype'] == 'PFF':
                                if Ctype == '3pt' or Ctype == '2pt' or Ctype == '3pt_all_t' or Ctype == '2pt_all_t':
                                    for gamma_list in data_info['gamma'][Mom_indx]:
                                        _gamma = f''

                                        for i in gamma_list:
                                            _gamma += f'{i:02d}'

                                        file_name += [file_name_Mom.replace(f'<gamma>', _gamma)]
                                    n = len(data_info['gamma'][0]) * len(data_info[kname])

                                elif Ctype == '2pt_mu_nu':
                                    for gamma_list in data_info['gamma_nu'][Mom_indx]:
                                        _gamma = f'{gamma_list[0]:02d}'

                                        file_name += [file_name_Mom.replace(f'<gamma>', _gamma)]
                                    
                                    n = len(data_info['gamma_nu'][0]) * len(data_info[kname])

                                elif Ctype == 'bubble':
                                    for gamma_list in data_info['gamma_mu'][Mom_indx]:
                                        _gamma = f'{gamma_list[0]:02d}'

                                        file_name += [file_name_Mom.replace(f'<gamma>', _gamma)]
                                    
                                    n = len(data_info['gamma_mu'][0]) * len(data_info[kname])
                            
                            else:
                                file_name += [file_init[j].replace(f'<{pname}>', _Mom)]
                                n = len(data_info[kname])

                elif kname == 'gamma' and data_info['dtype'] != 'PFF':
                    for gamma in data_info[kname]:
                        for gamma_list in gamma:
                            _gamma = f''

                            for i in gamma_list:
                                _gamma += f'{i}'

                            for j in range(np.prod(N)):
                                file_name += [file_init[j].replace(f'<{pname}>', _gamma)]

                    n = len(data_info[kname][0]) * len(data_info[kname])
                
                elif kname == 'gamma' and data_info['dtype'] == 'PFF':
                    continue

                else :
                    for i in data_info[kname]:
                        for j in range(np.prod(N)):
                            file_name += [file_init[j].replace(f'<{pname}>', f'{i}')]

                    n = len(data_info[kname])

                file_name = file_init = file_name[np.prod(N):]
                k += 1
                N = [n] + N

        position = adjust_duplicates_ordered([x - (5 - k + 1) for x in position])

        if check_file:
            import pathlib
            not_exist_name = []
            
            for file_path in file_name:
                path = pathlib.Path(file_path)

                if path.exists():
                    continue
                
                else:
                    not_exist_name += [file_path]
            
        file_name = np.asarray(file_name).reshape(N[:6]).transpose(position)
        
        if data_info['dtype'] == 'PFF':
            file_name_shape = list(file_name.shape)
            file_name_shape[1] = file_name_shape[0]//len(data_info['Mom'])
            file_name_shape[0] = len(data_info['Mom'])
            
            file_name = np.ascontiguousarray(file_name)
            file_name = file_name.reshape(file_name_shape)

            if Ctype == '2pt' or Ctype == '2pt_all_t' :
                file_name = file_name[:, :1]
        
        if check_file and not_exist_name != []:
            not_exist_file_indx = []
            for not_exist_file in not_exist_name:
                not_exist_file_indx += np.argwhere(file_name == not_exist_file).tolist()

            not_exist_file_indx = np.unique(np.asarray(not_exist_file_indx), axis = 0)
            # print(not_exist_file_indx[:5,:])
            not_exist_param = {}
            for i in range(6):
                # print(data_info[param_position(param = i)])
                # print(np.unique(not_exist_file_indx[:, i]).tolist())
                if data_info['dtype'] == 'PFF':
                    not_exist_param[param_position(param = i)] = [data_info[param_position(param = i)][x] for x in np.unique(not_exist_file_indx[:, i]).tolist()]
                
                elif data_info['dtype'] == 'PDF':
                    if param_position(param = i) != 'gamma':
                        not_exist_param[param_position(param = i)] = [data_info[param_position(param = i)][x] for x in np.unique(not_exist_file_indx[:, i]).tolist()]
                    
            raise ValueError(f"{Ctype} file param: {not_exist_param} not exist file test {not_exist_name[0]}")
        
        data_info[f'file_name_{Ctype}'] = file_name
        
    return data_info

# JackknifeJackknife: 
# 样本平均值是固定的，缩小样本的大小，减小方差；
# 但是不适用于样本量比较小的情况，会导致数据分析出现偏差；并且不适用于非线性的分析
def jcknf_sample(data):
    sum_dimension = np.asarray(np.shape(data))
    sum_dimension[-3] = 1
    n = data.shape[-3]
    data_sum = np.sum(data, axis = -3).reshape(sum_dimension)
    jcknf_sample = (data_sum - data)/(n-1)

    return jcknf_sample

def Jackknife(data_info:dict, name:Literal['data_2pt', 'data_3pt', 'data_2pt_all_t', 'data_3pt_all_t', 'data_2pt_mu_nu', 'bubble', 'data_2pt_bubble'] = 'data_2pt', only_sample:bool = False) -> dict:
    if name in data_info.keys():
        data = data_info[name]
        
        N0_indx = -3
        
        if name == 'data_2pt_mu_nu' or '_all_t' in name:
            N0_indx = -4

        N0 = data.shape[N0_indx]
        N1 = data.shape[N0_indx + 1]
        
        shape = list(data.shape)
        
        shape_mean_err = shape.copy()
        shape_mean_err[N0_indx] = 1
        
        shape_cov = shape.copy()
        shape_cov[N0_indx] = shape_cov[N0_indx + 1]
        
        N_sum = int(np.prod(shape[:N0_indx]))

        data_cov = np.zeros((N_sum, N1, N1, shape[-1]))
        
        data_sum = np.sum(data, axis = N0_indx, keepdims = True)
        data_mean = np.mean(data, axis = N0_indx, keepdims = True)
        
        data_info[f'{name}_sample'] = - (data - data_sum) / (N0 - 1) # solve the jackknife sample
        
        if only_sample == False:
            data_info[f'{name}_mean'] = np.mean(data, axis = N0_indx) # solve the mean value of data   
            data_info[f'{name}_err'] = np.sqrt(N0 - 1) * np.std(data_info[f'{name}_sample'], axis=N0_indx) # solve the standard deviation of data

            if N0_indx == -3:
                data_info[f'{name}_cov'] = np.mean(contract('...ab,...cb->...acb', data - data_mean, data - data_mean), axis = -4, keepdims = True)
                
            else:
                data_info[f'{name}_cov'] = 0
                
    
    else:
        print('analyse name not in the dict of data_info')

    return data_info

#Bootstrap: 随机抽取N次原样本中的M个数据，
def Bootstrap(data_info:dict, N:int = 0, M:int = 0, name:Literal['data_2pt', 'data_3pt'] = 'data_2pt') -> dict:
    if name in data_info.keys():
        data = data_info[name]

        N0 = data.shape[-3]
        N1 = data.shape[-2]
        
        if M == 0:
            M = N0 - 5
        if N == 0:
            N = N0 * 4
            
        if M > N1:
            raise print(f'M must less than N0. M{M}, N0{N0}.')
        
        if N > math.factorial(N0) / (math.factorial(N0 - M) * math.factorial(M)):
            raise print(f'N must less than C{N0}{M}. M{M}, N0{N0} and N0 must > 5')
        
        shape = list(data.shape)
        
        shape_mean_err = shape.copy()
        shape_mean_err[-3] = 1
        
        shape_cov = shape.copy()
        shape_cov[-3] = shape_cov[-2]
        
        N_sum = int(np.prod(shape[:-3]))

        data.reshape(N_sum, N0, N1, shape[-1])
        data_cov = np.zeros((N_sum, N1, N1, shape[-1]))
        
        accept_ratio = M / N0
        data_sample = np.zeros((N_sum, N, N1, shape[-1]))

        for i in range(N):
            random_array = np.random.random((N0))
            while ((random_array - accept_ratio) < 0).all():
                random_array = np.random.random((N0))
            data_sample[:, i] = np.sum(data[:, random_array > accept_ratio], axis = -3) / np.sum(random_array > accept_ratio)
        
        data_info[f'{name}_sample'] = data_sample
        
        data_info[f'{name}_mean'] = np.squeeze(np.mean(data_sample, axis = -3).reshape(shape_mean_err), axis=-3)
        data_info[f'{name}_err'] = np.squeeze(np.std(data_sample, axis = -3).reshape(shape_mean_err), axis=-3)
        
        for i in range(N_sum):
            for j in range(shape[-1]):
                data_cov[i, ..., j] = np.cov((data_info[f'{name}_sample'][..., j].reshape(N_sum, N0, N1))[i].T)

        data_info[f'{name}_cov'] = data_cov.reshape(shape_cov)
    
    else:
        print('analyse name not in the dict of data_info')

    return data_info

def flatten_generator(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten_generator(item)
        else:
            yield item

class ArraySlicer:
    """
    高级数组切片和赋值工具类
    使用 np.ix_ 构建索引网格
    """
    
    def __init__(self, array):
        """
        初始化
        
        Args:
            array: numpy数组
        """
        self.array = array
        self.ndim = array.ndim
        self.shape = array.shape

    def get_slices(self, dims, indices):
        """
        根据指定的维度和索引构建切片对象
        
        Args:
            dims: 维度列表
            indices: 对应维度的索引列表
            
        Returns:
            tuple: 切片对象
        """
        arr_dims = self.ndim
        arr_shape = self.shape
        
        if len(dims) != len(indices):
            raise ValueError("维度和索引列表长度必须相同")
        
        # 初始化每个维度的索引范围
        slices = [list(range(x)) for x in arr_shape]
        
        for dim, idx in zip(dims, indices):
            if dim >= arr_dims:
                raise ValueError(f"维度 {dim} 超出数组维度范围 (0-{arr_dims-1})")
            
            if isinstance(idx, (list, np.ndarray)):
                # 如果是列表或数组，直接使用
                slices[dim] = idx
            elif isinstance(idx, int):
                # 单个索引
                slices[dim] = [idx]
            elif idx == slice(None):
                # 保持原样
                slices[dim] = slices[dim]
            elif isinstance(idx, range):
                slices[dim] = list(idx)
            elif isinstance(idx, slice):
                # 将slice转换为range
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else arr_shape[dim]
                step = idx.step if idx.step is not None else 1
                slices[dim] = list(range(start, stop, step))
            else:
                raise ValueError(f"不支持的索引类型: {type(idx)}")
        
        return np.ix_(*slices)

    def slice(self, dims, indices):
        """
        切片提取
        
        Args:
            dims: 维度列表
            indices: 对应维度的索引列表
            
        Returns:
            ndarray: 切片后的数组
        """
        slices = self.get_slices(dims, indices)
        return self.array[slices]
    
    def assign(self, dims, indices, values, keep_dims:list = []):
        """
        赋值操作
        
        Args:
            dims: 维度列表
            indices: 对应维度的索引列表
            values: 要赋的值
            
        Returns:
            ndarray: 赋值后的数组（原地修改）
        """
        slices = self.get_slices(dims, indices)
        if isinstance(values, int):
            self.array[slices] = values

        else:
            if keep_dims == []:
                self.array[slices] = values.reshape(self.array[slices].shape)
            
            else:
                self.array[slices] = values.reshape([x if x_indx in keep_dims or (x_indx - self.ndim) in keep_dims else 1 for x_indx, x in enumerate(self.array[slices].shape)])

        return self.array
    
    def get_slice_shape(self, dims, indices):
        """
        获取切片后的形状
        
        Args:
            dims: 维度列表
            indices: 对应维度的索引列表
            
        Returns:
            tuple: 切片后的形状
        """
        slices = self.get_slices(dims, indices)
        
        # 创建一个虚拟数组来计算切片后的形状
        dummy_array = np.zeros(self.shape)
        sliced = dummy_array[slices]
        
        return sliced.shape
    
    def get_info(self):
        """获取数组信息"""
        return {
            'shape': self.array.shape,
            'ndim': self.array.ndim,
            'dtype': self.array.dtype
        }
        
def loop_tsrc(data, indx:list = [-2, -3], Boundary_Conditions: Literal['Periodic', 'Antiperiodic'] = 'Periodic', dtype:Literal['mean', 'sum'] = 'sum'):
    data_init = data.copy()
    data_shape = data_init.shape
    
    if len(indx) != 2:
        raise ValueError('indx must have 2 elements for tsrc, tsink')

    if data_shape[indx[0]] != data_shape[indx[1]]:
        raise ValueError('data[indx] must have the same size')
    
    Nt = data_shape[indx[0]]
    data_looped = np.zeros_like(np.sum(data_init, axis = indx[0], keepdims = True))
    
    if Boundary_Conditions == 'Antiperiodic':
        for tsrc in range(Nt):
            for tsink in range(Nt):
                data_slicer = ArraySlicer(data_init)

                if(tsink < tsrc):
                    data_init = data_slicer.assign(
                        dims = [indx[0], indx[1]], 
                        indices = [[tsrc], [tsink]], 
                        values = - 1.0 * data_slicer.slice(dims = indx, indices = [[tsrc], [tsink]])
                        )

    for tsrc in range(Nt):
        for tsink in range(Nt):

            data_looped_slicer = ArraySlicer(data_looped)
            data_slicer = ArraySlicer(data_init)

            data_looped = data_looped_slicer.assign(
                dims = [indx[1]], 
                indices = [(tsink - tsrc + Nt) % Nt], 
                values = data_slicer.slice(dims = indx, indices = [[tsrc], [tsink]]) + data_looped_slicer.slice(dims = [indx[1]], indices = [(tsink - tsrc + Nt) % Nt])
                )
            
            # print(data_looped)
    
    if dtype == 'sum':
        return data_looped.sum(indx[0])
    
    elif dtype == 'mean':
        return data_looped.mean(indx[0])

class data_analyse:
    def __init__(
        self,
        hadron:str,
        data_info:dict,

        over_all_coeff:float = 1.0,
        
        data_info_min:dict = False,
        time_fold:bool = False,
        link_fold:bool = False,
        conserved:bool = False,
        statistics_type:Literal['jackknife', 'bootstrap'] = 'jackknife'

        ):
        
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        self.fm2GeV = 0.1973
        self.hadron = hadron
        self.time_fold = time_fold
        self.conserved = conserved
        self.link_fold = link_fold
        self.over_all_coeff = over_all_coeff
        
        for data_indx in data_info.keys():
            if data_info[data_indx]['read_type'][0] == 'both':
                read_list = ['3pt', '2pt', '2pt_mu_nu', 'bubble']

            else:
                read_list = data_info[data_indx]['read_type']
            
            for name in read_list:
                data_info[data_indx] = self.read(data_info[data_indx], name = name)

                if statistics_type == 'jackknife':
                    data_info[data_indx] = Jackknife(data_info[data_indx], name = f'data_{name}')
                
            if data_info_min != False:
                for name in read_list:
                    data_info_min[data_indx] = self.read(data_info_min[data_indx], name = name)

                    if statistics_type == 'jackknife':
                        data_info_min[data_indx] = Jackknife(data_info_min[data_indx], name = f'data_{name}')
            
            conf_list = data_info[data_indx]['conf']
            conf_list_matrix = comm.gather(conf_list, root = 0)
            
            if rank == 0:
                data_info[data_indx]['conf'] = list(flatten_generator(conf_list_matrix))
                
            for name in read_list:
                data = np.ascontiguousarray(data_info[data_indx][f'data_{name}'].copy())
                data = np.moveaxis(data, param_position('conf'), 0)
                
                data_matrix = comm.gather(data, root = 0)

                if rank == 0:
                    data_shape = list(data.shape)
                    data_shape[0] = len(data_info[data_indx]['conf'])
                    
                    data_matrix = np.asarray(list(flatten_generator([x.tolist() for x in data_matrix]))).reshape(data_shape)
                    data_matrix = np.moveaxis(data_matrix, 0, param_position('conf'))
                    
                    data_info[data_indx][f'data_{name}'] = data_matrix.copy()
                    
                    if statistics_type == 'jackknife':
                        data_info[data_indx] = Jackknife(data_info[data_indx], name = f'data_{name}')
                
            if data_info_min != False:
                for name in read_list:
                    data = np.ascontiguousarray(data_info_min[data_indx][f'data_{name}'].copy())
                    data = np.moveaxis(data, param_position('conf'), 0)
                    data_matrix = comm.gather(data, root = 0)

                    if rank == 0:
                        data_shape = list(data.shape)
                        data_shape[0] = len(data_info[data_indx]['conf'])
                        data_matrix = np.asarray(list(flatten_generator(data_matrix))).reshape(data_shape)
                        data_info_min[data_indx][f'data_{name}'] = data_matrix.copy()
                        
                        if statistics_type == 'jackknife':
                            data_info_min[data_indx] = Jackknife(data_info_min[data_indx], name = f'data_{name}')

            if time_fold:
                data_info[data_indx]['Nt'] = data_info[data_indx]['Nt'] // 2 +1
        
        if rank == 0:
            for data_indx in data_info.keys():
                if '2pt' in data_info[data_indx]['read_type']:
                    data_info[data_indx] = self.meff_2pt(data_info = data_info[data_indx]).copy()

                if data_info[data_indx]['dtype'] == 'PFF':
                    # self.pi2gamma()
                    continue

                else:
                    continue
                    # if '3pt' in read_list:
                    #     data_info[data_indx] = self.PDF(data_info = data_info[data_indx], link_fold = link_fold).copy()
                    #     data_info[data_indx] = self.link_analyse(data_info = data_info[data_indx]).copy()
                    
                    # if ('2pt_mu_nu' in read_list) and ('bubble' in read_list):
                    #     data_info = self.dis_connect(data_info = data_info)

            self.data_info = data_info

    def read(self, data_info, name:Literal['3pt', '2pt', '2pt_mu_nu', 'bubble']):
        if name == '2pt':
            readed = np.zeros((len(data_info['Mom']), 1, len(data_info['Nev']), 1, 1, len(data_info['conf']), data_info['Nt'], 3), dtype = float)
            
        elif name == '2pt_all_t':
            readed = np.zeros((len(data_info['Mom']), 1, len(data_info['Nev']), 1, 1, len(data_info['conf']), data_info['Nt'], data_info['Nt'], 3), dtype = float)

        elif name == '3pt':
            readed = np.zeros((len(data_info['Mom']), len(data_info['gamma'][0]), len(data_info['Nev']), len(data_info['tsep']), len(data_info['link_list']), len(data_info['conf']), data_info['Nt'], 3), dtype = float)
        
        elif name == '3pt_all_t':
            readed = np.zeros((len(data_info['Mom']), len(data_info['gamma'][0]), len(data_info['Nev']), len(data_info['tsep']), len(data_info['link_list']), len(data_info['conf']), data_info['Nt'], data_info['Nt'], 3), dtype = float)
            
        elif name == '2pt_mu_nu':
            readed = np.zeros((len(data_info['Mom']), len(data_info['gamma_nu'][0]), len(data_info['Nev']), 1, 1, len(data_info['conf']), data_info['Nt'], data_info['Nt'], 3), dtype = float)

        elif name == 'bubble':
            readed = np.zeros((len(data_info['Mom']), len(data_info['gamma_mu'][0]), len(data_info['Nev']), 1, len(data_info['link_list']), len(data_info['conf']), data_info['Nt'], 3), dtype = float)
        
        else:
            raise ValueError(f'{name} type don\'t support.')

        Nt = data_info['Nt']

        if '.dat' in data_info[f'file_{name}']:
            for P in range(readed.shape[0]):
                for gamma in range(readed.shape[1]):
                    for Nev in range(readed.shape[2]):
                        for tsep in range(readed.shape[3]):
                            for link in range(readed.shape[4]):
                                for conf in range(readed.shape[5]):
                                    data = np.loadtxt(data_info[f'file_name_{name}'][P, gamma, Nev, tsep, link, conf], skiprows = 1, delimiter = ' ')
                                    for i in range(3):
                                        readed[P, gamma, Nev, tsep, link, conf, :, i] = data[:, i]


        if '.iog' in data_info[f'file_{name}']:
            from iog_reader.iog_reader import iog_read

            intrptr = ['ID']
            for P in range(readed.shape[0]):
                for gamma in range(readed.shape[1]):
                    for Nev in range(readed.shape[2]):
                        for tsep in range(readed.shape[3]):
                            for link in range(1):
                                for conf in range(readed.shape[5]):
                                    data = iog_read(data_info[f'file_name_{name}'][P, gamma, Nev, tsep, link, conf], intrptr)
                                    if name == '2pt' or name == '2pt_mu_nu' or name == 'bubble':
                                        readed[P, gamma, Nev, tsep, link, conf, :, 1] = data['Re'].to_numpy()
                                        readed[P, gamma, Nev, tsep, link, conf, :, 2] = data['Im'].to_numpy()

                                    elif name == '3pt':
                                        N_link = len(data_info['link_list'])
                                        link_max = N_link//2
                                        
                                        readed[P, gamma, Nev, tsep, :, conf, :, 1] = np.append(
                                            data['Re'].to_numpy().reshape(N_link, -1, Nt)[link_max+1:, 8][::-1], 
                                            data['Re'].to_numpy().reshape(N_link, -1, Nt)[:link_max+1, 8], 
                                            axis=0
                                            )
                                        
                                        readed[P, gamma, Nev, tsep, :, conf, :, 2] = np.append(
                                            data['Im'].to_numpy().reshape(N_link, -1, Nt)[link_max+1:, 8][::-1], 
                                            data['Im'].to_numpy().reshape(N_link, -1, Nt)[:link_max+1, 8], 
                                            axis=0
                                            )

        if '.npy' in data_info[f'file_{name}']:
            for P in range(readed.shape[0]):
                for gamma in range(readed.shape[1]):
                    for Nev in range(readed.shape[2]):
                        for tsep in range(readed.shape[3]):
                            for link in range(readed.shape[4]):
                                for conf in range(readed.shape[5]):
                                    data = np.load(data_info[f'file_name_{name}'][P, gamma, Nev, tsep, link, conf])

                                    readed[P, gamma, Nev, tsep, link, conf, ..., 1] = data.real
                                    readed[P, gamma, Nev, tsep, link, conf, ..., 2] = data.imag
        if name == '2pt':
            if self.time_fold == True:
                readed = readed[..., :Nt//2+1, :]

        elif name == '3pt':
            if self.time_fold == True:
                if self.conserved == True:
                    readed_A = np.roll(readed, shift=-1, axis=-2)
                    readed_B = readed_A[...,0:Nt//2 + 1,:] - readed_A[...,Nt//2 - 1:,:][...,::-1,:]
                    readed[...,:Nt//2,:] = readed_B[...,:Nt//2,:]
                    readed[...,Nt//2,:] = readed[...,Nt//2,:] + readed[...,Nt//2 + 1,:]

                else:
                    readed = readed[...,1:Nt//2+2,:] - readed[...,Nt//2-1:,:][...,::-1,:]
            else:
                if self.conserved == True:
                    readed = np.roll(readed, shift=-1, axis=-2)

        data_info[f'data_{name}'] = readed[..., 1:] * self.over_all_coeff
        
        del data_info[f'file_name_{name}']
        
        return data_info
    
    def meff(self, data_info:dict, type:Literal['2pt_all_t', '2pt'] = None):
        Ncnfg = len(data_info['conf'])
        if type == None:
            type = [x for x in data_info['read_type'] if '2pt' in x][-1]
            
        if type == '2pt_all_t':
            C2pt_sample = data_info['data_2pt_all_t_sample']
            
            for t in range(data_info['Nt']):
                C2pt_sample[..., t, :, :] = np.roll(C2pt_sample[..., t, :, :], shift = -t, axis = -2)
                
        elif type == '2pt':
            C2pt_sample = data_info['data_2pt_sample']
        
        meff_sample = np.zeros_like(C2pt_sample)
        
        if data_info['meff_type'] == 'log':    
            meff_sample[..., :data_info['Nt'] - 1, :] = np.array(np.log(np.real(C2pt_sample[..., :-1, :])/np.real(C2pt_sample[..., 1:, :]))) * (self.fm2GeV / data_info['alttc'])
            
        else:
            meff_sample[..., :data_info['Nt'] - 2, :] = np.arccosh((C2pt_sample[..., 2:, :] + C2pt_sample[..., :-2, :])/(2 * C2pt_sample[..., 1:-1, :])) * (self.fm2GeV / data_info['alttc'])
            
        if type == '2pt_all_t':
            for t in range(data_info['Nt']):
                meff_sample[..., t, :, :] = np.roll(meff_sample[..., t, :, :], shift = t, axis = -2)
                
            meff_mean = np.mean(meff_sample, axis = -4)
            meff_err = np.std(meff_sample, axis = -4) * np.sqrt(Ncnfg - 1)

        elif type == '2pt':
            meff_mean = np.mean(meff_sample, axis = -3)
            meff_err = np.std(meff_sample, axis = -3) * np.sqrt(Ncnfg - 1)
        
        data_info['meff_2pt_sample'] = meff_sample    
        data_info['meff_2pt_mean'] = meff_mean
        data_info['meff_2pt_err'] = meff_err
        
        return data_info
    
    def meff_2pt(self, data_info:dict): # data:1, P, ENV, 1, 1, Ncnfg, Nt, dtype=complex
        C2pt_mean = data_info['data_2pt_mean']
        C2pt_jcknf = data_info['data_2pt_sample']

        Ncnfg = len(data_info['conf'])
        T_hlf = data_info['Nt']//2

        if data_info['meff_type'] == 'log':
            data_mean = np.array(np.log(np.real(C2pt_mean[..., :-1, :])/np.real(C2pt_mean[..., 1:, :]))) * (self.fm2GeV / data_info['alttc'])
            data_log_sample = np.array(np.log(np.real(C2pt_jcknf[..., :-1, :])/np.real(C2pt_jcknf[..., 1:, :]))) * (self.fm2GeV / data_info['alttc'])
            data_err = np.std(data_log_sample, axis=-3) * np.sqrt(Ncnfg - 1)
            # data_err = np.sqrt(np.sum((data_log_sample - data_mean)**2, axis=-2) * (Ncnfg-1) / Ncnfg)
        
        elif data_info['meff_type'] == 'cosh':
            data_mean = np.arccosh((C2pt_mean[..., 2:, :] + C2pt_mean[..., :-2, :])/(2 * C2pt_mean[..., 1:-1, :])) * (self.fm2GeV / data_info['alttc'])
            data_cosh_sample = np.arccosh((C2pt_jcknf[..., 2:, :] + C2pt_jcknf[..., :-2, :])/(2 * C2pt_jcknf[..., 1:-1, :])) * (self.fm2GeV / data_info['alttc'])
            
            # C2pt_jcknf = C2pt_jcknf.astype(np.float64)
            # t_ary = np.array(range(self.Nt-1))
            # ini = 0.2*np.ones_like(t_ary)
            
            # def eff_mass_eqn(c2pt):
            #     return lambda E0: (c2pt[:-1]/c2pt[1:]) - np.cosh(E0*(T_hlf-t_ary)) / np.cosh(E0*(T_hlf-(t_ary+1)))
            
            # def fndroot(eqnf,ini):
            #     sol = fsolve(eqnf, ini, xtol=1e-5)
            #     return sol

            # data_cosh_sample = np.array([fndroot(eff_mass_eqn(c2pt),ini) for c2pt in C2pt_jcknf[:,:]])

            # data_mean = np.mean(data_cosh_sample,axis=-2)

            data_err = np.sqrt(Ncnfg - 1) * np.std(data_cosh_sample, axis = -3)

        data_info['meff_2pt_mean'] = data_mean
        data_info['meff_2pt_err'] = data_err

        return data_info
    
    def dis_connect(self, data_info:dict, dtype:Literal['Im', 'Re', 'Complex'] = 'Im', conf_list:list = [], compensation_matrix = 0):
        if dtype == 'Re':
            axis = 0

        elif dtype == 'Im':
            axis = 1

        if conf_list == []:
            conf_list = [x_indx for x_indx, x in enumerate(data_info['conf'])]

        Nt = data_info['Nt']
        tsep = data_info['tsep']

        if ('2pt_mu_nu' in data_info['read_type']) and ('bubble' in data_info['read_type']):
            if dtype == 'Complex':
                data_2pt_mu_nu = data_info['data_2pt_mu_nu'][..., 0].copy() + 1j * data_info['data_2pt_mu_nu'][..., 1].copy()
                data_bubble = data_info['data_bubble'][..., 0].copy() + 1j * data_info['data_bubble'][..., 1].copy() 
                arr_dtype = complex
            else:    
                data_2pt_mu_nu = data_info['data_2pt_mu_nu'][..., axis].copy()
                data_bubble = data_info['data_bubble'][..., 0].copy()
                arr_dtype = float
                
            data_2pt_bubble_shape = list(data_info['data_2pt_mu_nu'].shape)
            data_2pt_bubble_shape[param_position(param = 'tsep')] = len(tsep)
            data_2pt_bubble_shape[param_position(param = 'link_list')] = len(data_info['link_list'])
            data_2pt_bubble_matrix = np.zeros(data_2pt_bubble_shape, dtype = float)
            
            data_2pt_mu_nu_shape = list(data_2pt_mu_nu.shape)
            data_2pt_mu_nu_shape[param_position(param = 'link_list')] = len(data_info['link_list'])
            _data_bubble = np.zeros(data_2pt_mu_nu_shape, dtype = arr_dtype)

            data_2pt_mu_nu_mean = np.mean(data_2pt_mu_nu, axis = -3, keepdims = True)
            data_bubble_mean = np.mean(data_bubble, axis = -2, keepdims = True)

            _data_2pt_mu_nu = data_2pt_mu_nu - data_2pt_mu_nu_mean

            if type(compensation_matrix) != int:
                for t in range(Nt):
                    compensation_matrix[..., t, :, :] = np.roll(compensation_matrix[..., t, :, :], axis = -2, shift = -t)
                
            for t in range(Nt):
                _data_2pt_mu_nu[..., t, :] = np.roll(_data_2pt_mu_nu[..., t, :], axis = -1, shift = -t)
            
            for t in range(Nt):
                _data_bubble[..., t, :] = np.roll(data_bubble - data_bubble_mean, axis = -1, shift = -t)

            # _data_bubble = np.roll(_data_bubble, axis = 0, shift = 3)
            if data_info['dtype'] == 'PFF':
                _data_2pt_mu_nu = np.roll(_data_2pt_mu_nu, axis = 0, shift = len( data_info['Mom']) // 2)

            data_2pt_bubble_1 = contract('MGETLCmn,MGETLCmb->MGETLCmnb', _data_2pt_mu_nu[..., tsep], _data_bubble)
            
            if data_info['dtype'] == 'PFF':
                # _data_bubble = np.roll(_data_bubble, axis = 0, shift = 3)
                data_2pt_bubble_2 = contract('MGETLCmn,MGETLCmb->MGETLCmbn', _data_2pt_mu_nu, _data_bubble[..., tsep])
                
                for t_sep_indx, t_sep in enumerate(tsep):
                    if dtype == 'Complex':
                        data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx+1), ..., 0] = (
                            np.append(data_2pt_bubble_1[..., t_sep_indx:(t_sep_indx+1), :t_sep] , data_2pt_bubble_2[..., t_sep_indx:(t_sep_indx+1), t_sep:], axis = -1)
                            ).transpose(0,1,2,7,4,5,6,3,8).reshape(data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx+1), ..., 0].shape).real

                        data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx+1), ..., 1] = (
                            np.append(data_2pt_bubble_1[..., t_sep_indx:(t_sep_indx+1), :t_sep] , data_2pt_bubble_2[..., t_sep_indx:(t_sep_indx+1), t_sep:], axis = -1)
                            ).transpose(0,1,2,7,4,5,6,3,8).reshape(data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx+1), ..., 0].shape).imag

                        if type(compensation_matrix) != int:
                            data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx + 1), ..., t_sep, :] = compensation_matrix[..., t_sep, :]

                    else:
                        data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx+1), ..., axis] = (
                            np.append(data_2pt_bubble_1[..., t_sep_indx:(t_sep_indx+1), :t_sep] , data_2pt_bubble_2[..., t_sep_indx:(t_sep_indx+1), t_sep:], axis = -1)
                            ).transpose(0,1,2,7,4,5,6,3,8).reshape(data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx+1), ..., 0].shape)

                        if type(compensation_matrix) != int:
                            data_2pt_bubble_matrix[:, :, :, t_sep_indx:(t_sep_indx + 1), ..., t_sep, axis] = compensation_matrix[..., t_sep, axis]

            elif data_info['dtype'] == 'PDF':
                link_max = (len(data_info['link_list']) - 1)//2
                
                if dtype == 'Complex':
                    data_2pt_bubble_matrix[..., 0] = (data_2pt_bubble_1).transpose(0,1,2,7,4,5,6,3,8).reshape(data_2pt_bubble_matrix[..., 0].shape).real
                    data_2pt_bubble_matrix[..., 1] = (data_2pt_bubble_1).transpose(0,1,2,7,4,5,6,3,8).reshape(data_2pt_bubble_matrix[..., 0].shape).imag

                else:
                    data_2pt_bubble_matrix[..., axis] = (data_2pt_bubble_1).transpose(0,1,2,7,4,5,6,3,8).reshape(data_2pt_bubble_matrix[..., 0].shape)
                
                if self.link_fold:
                    data_2pt_bubble_matrix = np.append(data_2pt_bubble_matrix[:, :, :, :, link_max:link_max+1], (data_2pt_bubble_matrix[:, :, :, :, link_max + 1:] - data_2pt_bubble_matrix[:, :, :, :, 0:link_max][:, :, :, :, ::-1])/2, axis = 4)
                    
            data_info['data_2pt_bubble'] = data_2pt_bubble_matrix.sum(-3)[..., conf_list, :, :]

            data_info = Jackknife(data_info, name = 'data_2pt_bubble', only_sample = True)

        return data_info

    def PDF(self, data_info:dict, link_fold:bool = False, dtype:Literal['3pt', '2pt_bubble'] = '3pt'): #P, ENV, tsep, link, Ncnfg, Nt, dtype=complex
        
        # the number of data
        if dtype == '3pt':
            data_3pt = data_info['data_3pt']
            data_3pt_sample = data_info['data_3pt_sample']
        
        elif dtype == '2pt_bubble':
            data_3pt = data_info['data_2pt_bubble']
            data_3pt_sample = data_info['data_2pt_bubble_sample']
            
        N_link = len(data_info['link_list'])
        link_max = int(len(data_info['link_list'])//2)
        
        data_2pt_sample = data_info['data_2pt_sample']

        if link_fold == True:
            N_link = link_max + 1
            data_3pt_sample_fold = (
                data_3pt_sample[..., link_max:, :, :, :] + data_3pt_sample[..., :link_max + 1, :, :, :][..., ::-1, :, :, :]
                )/2
        
        else:
            data_3pt_sample_fold = data_3pt_sample

        N_P = data_3pt.shape[0]; N_gamma = data_3pt.shape[1]; N_Nev = data_3pt.shape[2]; N_tsep = data_3pt.shape[3]; Ncnfg = data_3pt.shape[-3]; Nt = data_info['Nt']

        tsep_list = data_info['tsep']

        PDF_3pt_2pt_mean = np.zeros((N_P, N_gamma, N_Nev, N_tsep, N_link, Nt, 2), dtype=float)
        PDF_3pt_2pt_sample = np.zeros(( N_P, N_gamma, N_Nev, N_tsep, N_link, Ncnfg, Nt, 2), dtype=float)
        # PDF_3pt_2pt = np.zeros((N_P, N_gamma, N_Nev, N_tsep, N_link, Ncnfg, Nt, 2), dtype=float)
        PDF_3pt_2pt_cov = np.zeros((N_P, N_gamma, N_Nev, N_tsep, N_link, Nt, Nt), dtype=float)

        # the 3pt/2pt part
        for tsep_indx in range(N_tsep):
            PDF_3pt_2pt_sample[..., tsep_indx, :, :, :, :] = data_3pt_sample_fold[..., tsep_indx, :, :, :, :] / data_2pt_sample[...,tsep_list[tsep_indx], :].reshape(N_P, -1, N_Nev, 1, Ncnfg, 1, 2)
            PDF_3pt_2pt_mean[..., tsep_indx, :, :, :] = np.mean(PDF_3pt_2pt_sample[..., tsep_indx, :, :, :, :], axis=-3)

            #ratio cov
            # cov_mean_dimension = np.asarray(np.shape(PDF_3pt_2pt_sample))
            # cov_mean_dimension[-1] = 1
            # cov_matrix = PDF_3pt_2pt_sample[:,:,:,tsep_indx,:,:,:tsep_array[tsep_indx]+1] - np.mean(PDF_3pt_2pt_sample[:,:,:,tsep_indx,:,:,:tsep_array[tsep_indx]+1],axis=-1).reshape(cov_mean_dimension)
            # PDF_3pt_2pt_cov[:,:,:,tsep_indx,:,:tsep_array[tsep_indx]+1,:tsep_array[tsep_indx]+1] = (1/(Nt-1)) * np.transpose(cov_matrix,(0,1,2,3,5,4)) @ cov_matrix
            
        #ratio_err
        # data_info['data_PDF'] = PDF_3pt_2pt
        if dtype == '3pt':
            data_info['data_PDF_sample'] = PDF_3pt_2pt_sample
            data_info['data_PDF_mean'] = PDF_3pt_2pt_mean
            data_info['data_PDF_err'] = np.std(PDF_3pt_2pt_sample, axis = -3) * np.sqrt(Ncnfg - 1)
            data_info['data_PDF_cov'] = PDF_3pt_2pt_cov
            
        elif dtype == '2pt_bubble':
            data_info['data_2pt_bubble_PDF_sample'] = PDF_3pt_2pt_sample
            data_info['data_2pt_bubble_PDF_mean'] = PDF_3pt_2pt_mean
            data_info['data_2pt_bubble_PDF_err'] = np.std(PDF_3pt_2pt_sample, axis = -3) * np.sqrt(Ncnfg - 1)
            data_info['data_2pt_bubble_PDF_cov'] = PDF_3pt_2pt_cov

        return data_info

    def link_analyse(self, data_info:dict):
        data_PDF_sample = data_info['data_PDF_sample']

        N_P = data_PDF_sample.shape[0]; N_gamma = data_PDF_sample.shape[1]; N_Nev = data_PDF_sample.shape[2]; N_tsep = data_PDF_sample.shape[3]; N_link = data_PDF_sample.shape[4]
        Ncnfg = data_PDF_sample.shape[-3]; Nt = data_info['Nt']

        link_mean = np.zeros((N_P, N_gamma, N_Nev, N_tsep, N_link, 2), dtype=float)
        link_err = np.zeros((N_P, N_gamma, N_Nev, N_tsep, N_link, 2), dtype=float)
        tsep_list = data_info['tsep']
        
        for tsep_indx in range(N_tsep):
            
            Y = gv.gvar(
                data_info['data_PDF_mean'][..., tsep_indx, :, (tsep_list[tsep_indx]//2), :], data_info['data_PDF_err'][..., tsep_indx, :, (tsep_list[tsep_indx]//2), :]
                )
            
            link_mean[..., tsep_indx, :, :] = gv.mean(Y)
            link_err[..., tsep_indx, :, :] = gv.sdev(Y)
        
        data_info['data_link_mean'] = link_mean
        data_info['data_link_err'] = link_err

        return data_info