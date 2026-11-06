#ifndef PYQCD_EINSUM_MPI_H
#define PYQCD_EINSUM_MPI_H

#include "einsum_runtime.h"

#if defined(PYQCD_EINSUM_ENABLE_MPI)

namespace pyqcd_einsum {

int execute_parsed_contract(const ExecutionPlan& execution, long long tensors_address,
                            long long out_address);

}  // namespace pyqcd_einsum

#endif

#endif
