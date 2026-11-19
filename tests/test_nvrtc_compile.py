"""Compile the real kernels with NVRTC, including on CPU-only CI workers.

Optional dependency: pip install '.[dev,nvrtc-test]'. No driver, GPU, CMake,
nvcc or host compiler is used by these tests.
"""
import ctypes as C
import importlib.util
import os
from pathlib import Path
import pytest
from dupire_identify.backend import cuda_source


class NvrtcCompileError(RuntimeError):
    pass


@pytest.fixture(scope="module")
def compiler():
    override=os.environ.get("DUPIRE_NVRTC_LIBRARY")
    paths=[]
    if override:
        path=Path(override)
        if not path.is_file():pytest.fail(f"NVRTC library does not exist: {path}")
        paths=[path]
    else:
        for module in ("nvidia.cuda_nvrtc","nvidia.cu13"):
            try:spec=importlib.util.find_spec(module)
            except ModuleNotFoundError:spec=None
            if spec:
                for root in spec.submodule_search_locations or ():
                    paths.extend(Path(root).glob("lib/libnvrtc.so.*"))
                    paths.extend(Path(root).glob("bin/nvrtc64_*.dll"))
    paths=[p for p in paths if ".alt." not in p.name]
    if not paths:
        if os.environ.get("DUPIRE_REQUIRE_NVRTC")=="1":
            pytest.fail("NVRTC compilation is required; install .[nvrtc-test]")
        pytest.skip("Optional NVRTC runtime not installed (no GPU required)")
    path=sorted(paths)[0]
    dll_context=os.add_dll_directory(str(path.parent)) if os.name=="nt" else None
    lib=C.CDLL(str(path))
    prog_t=C.c_void_p
    lib.nvrtcCreateProgram.argtypes=[C.POINTER(prog_t),C.c_char_p,C.c_char_p,C.c_int,
                                    C.POINTER(C.c_char_p),C.POINTER(C.c_char_p)]
    lib.nvrtcCompileProgram.argtypes=[prog_t,C.c_int,C.POINTER(C.c_char_p)]
    lib.nvrtcGetProgramLogSize.argtypes=[prog_t,C.POINTER(C.c_size_t)]
    lib.nvrtcGetProgramLog.argtypes=[prog_t,C.c_void_p]
    lib.nvrtcGetPTXSize.argtypes=[prog_t,C.POINTER(C.c_size_t)]
    lib.nvrtcGetPTX.argtypes=[prog_t,C.c_void_p]
    lib.nvrtcDestroyProgram.argtypes=[C.POINTER(prog_t)]
    lib.nvrtcVersion.argtypes=[C.POINTER(C.c_int),C.POINTER(C.c_int)]
    for name in ("nvrtcCreateProgram","nvrtcCompileProgram","nvrtcGetProgramLogSize",
                 "nvrtcGetProgramLog","nvrtcGetPTXSize","nvrtcGetPTX",
                 "nvrtcDestroyProgram","nvrtcVersion"):
        getattr(lib,name).restype=C.c_int
    def check(result):
        if result:raise RuntimeError(f"NVRTC API status={result}")
    def compile_source(source,options=()):
        program=prog_t()
        # Zero supplied headers and no include path: reproduce pip-only installs.
        check(lib.nvrtcCreateProgram(C.byref(program),source.encode("utf-8"),
                                    b"dupire_identify.cu",0,None,None))
        try:
            opts=[b"--std=c++11",b"--gpu-architecture=compute_86"]+[s.encode('ascii') for s in options]
            array=(C.c_char_p*len(opts))(*opts)
            result=lib.nvrtcCompileProgram(program,len(opts),array)
            size=C.c_size_t()
            check(lib.nvrtcGetProgramLogSize(program,C.byref(size)))
            log=C.create_string_buffer(max(1,size.value))
            check(lib.nvrtcGetProgramLog(program,log))
            if result:raise NvrtcCompileError(f"NVRTC status={result}: {log.value.decode(errors='replace')}")
            check(lib.nvrtcGetPTXSize(program,C.byref(size)))
            ptx=C.create_string_buffer(size.value)
            check(lib.nvrtcGetPTX(program,ptx))
            return ptx.value.decode("utf-8")
        finally:
            check(lib.nvrtcDestroyProgram(C.byref(program)))
    try:
        yield compile_source
    finally:
        if dll_context:dll_context.close()


def test_production_kernels_compile_without_host_headers(compiler):
    ptx=compiler(cuda_source())
    assert ".entry price_kernel" in ptx
    assert ".entry black_kernel" in ptx


def test_new_batched_diffusion_kernel_compiles_without_headers(compiler):
    from dupire_identify.fd_acceleration import cuda_source as fd_source
    assert '.entry fd_batch_kernel' in compiler(fd_source(),('--fmad=false','--prec-div=true','--prec-sqrt=true'))


def test_unconditional_host_math_header_reproduces_reported_error(compiler):
    # Control: the original unconditional include must reproduce the failure.
    broken=cuda_source().replace("#ifndef __CUDACC_RTC__","#if 1",1)
    with pytest.raises(NvrtcCompileError,match="math.h"):
        compiler(broken)
