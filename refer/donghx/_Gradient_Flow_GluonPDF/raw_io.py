"""Read-only mapping of uncompressed NPY members; no extraction/scratch copy."""
import struct
import zipfile
import numpy as np


def member(path,name):
    with zipfile.ZipFile(path) as archive:
        info=archive.getinfo(name+'.npy')
        if info.compress_type!=zipfile.ZIP_STORED:
            with np.load(path,allow_pickle=False) as d:return np.asarray(d[name])
        with open(path,'rb') as f:
            f.seek(info.header_offset)
            header=struct.unpack('<IHHHHHIIIHH',f.read(30))
            if header[0]!=0x04034b50:raise ValueError('invalid ZIP local header')
            f.seek(header[-2]+header[-1],1)
            start=f.tell()
            version=np.lib.format.read_magic(f)
            shape,order,dtype=np.lib.format._read_array_header(f,version)
            offset=f.tell()
            if dtype.hasobject:raise ValueError('object payload not allowed')
            count=int(np.prod(shape))
            if offset-start+count*dtype.itemsize!=info.file_size:
                raise ValueError('truncated/incompatible embedded NPY')
        return np.memmap(path,mode='r',dtype=dtype,offset=offset,shape=shape,order='F' if order else 'C')


def raw_headers(path):
    expected={'corr_2pt':(4,3,64,64),
        'corr_3pt_u':(2,4,3,21,4,4,64,64),'corr_3pt_d':(2,4,3,21,4,4,64,64)}
    for name,shape in expected.items():
        value=member(path,name)
        if value.shape!=shape or value.dtype!=np.dtype('complex128'):
            raise ValueError((path,name,value.shape,str(value.dtype)))
        del value
