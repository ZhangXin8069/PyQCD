import sys
import mpi4py
mpi4py.rc.thread_level = sys.argv[1]
from mpi4py import MPI
import numpy as np
c = MPI.COMM_WORLD
assert c.size == 2
x = np.full(32, c.rank, dtype='f8'); y = np.empty_like(x)
c.Allreduce(x, y)
assert np.all(y == 1)
c.Barrier()
if c.rank == 0: print('PASS thread=%d lib=%s' % (MPI.Query_thread(), MPI.Get_library_version().splitlines()[0]), flush=True)
