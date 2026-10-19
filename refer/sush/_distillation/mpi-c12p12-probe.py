"""Small MPI/CuPy environment probe; does not read production data."""
import os
import sys
import mpi4py

mpi4py.rc.thread_level = sys.argv[1]
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.rank
assert comm.size == 2
send = np.arange(16, dtype=np.float64) + rank
recv = np.empty_like(send)
comm.Allreduce(send, recv)
np.testing.assert_array_equal(recv, 2 * np.arange(16) + 1)
comm.Sendrecv(send, dest=1-rank, recvbuf=recv, source=1-rank)
np.testing.assert_array_equal(recv, np.arange(16) + 1-rank)
import cupy as cp
local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
cp.cuda.Device(local_rank).use()
x = cp.arange(16, dtype=cp.float64)
assert float(cp.sum(x).get()) == 120.0
print(f'PASS rank={rank} thread={MPI.Query_thread()} gpu={local_rank} '
      f'cupy={cp.__version__} runtime={cp.cuda.runtime.runtimeGetVersion()}', flush=True)
if rank == 0:
    print(MPI.Get_library_version().strip('\x00'), flush=True)
comm.Barrier()
