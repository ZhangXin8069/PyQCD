from libc.stdint cimport int64_t


cdef extern from "pyqcdeinsum.h":
    enum:
        QCD_STATUS_OK
        QCD_STATUS_INVALID
        QCD_STATUS_MEMORY
        QCD_STATUS_INTERNAL

    int applyPlanQcd(int64_t set_ptrs, int64_t params) noexcept nogil
    int applyContractQcd(
        int64_t plan,
        int64_t tensors,
        int64_t out,
        int64_t params,
    ) noexcept nogil
    int applyCompileQcd(
        int64_t plan,
        int64_t params,
        int64_t out_context,
    ) noexcept nogil
    int applyCompiledQcd(
        int64_t context,
        int64_t tensors,
        int64_t out,
        int64_t params,
    ) noexcept nogil
    void applyDestroyQcd(int64_t context) noexcept nogil
    void applyInitQcd(int64_t set_ptrs, int64_t params) noexcept nogil
    void applyEndQcd(int64_t set_ptrs, int64_t params) noexcept nogil
