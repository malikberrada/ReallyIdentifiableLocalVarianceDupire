r"""One-time repair of an already installed 0.1.0, without reinstalling CUDA.

Run with the affected interpreter:
  .\.venv\Scripts\python.exe scripts\repair_cuda_nvrtc.py
Then run: python -m dupire_identify doctor --backend cuda
"""
from pathlib import Path
import dupire_identify


def repair(header):
    source=header.read_text(encoding="utf-8")
    guard="#ifndef __CUDACC_RTC__"
    if guard in source:
        print(f"Already patched: {header}")
        return False
    old="#include <math.h>"
    if source.count(old)!=1:
        raise RuntimeError(f"Expected exactly one math.h include in {header}")
    updated=source.replace(old,guard+"\n"+old+"\n#endif",1)
    backup=header.with_name(header.name+".before-nvrtc-fix")
    if not backup.exists():backup.write_bytes(header.read_bytes())
    header.write_text(updated,encoding="utf-8")
    print(f"Patched: {header}")
    return True


if __name__=="__main__":
    installed=Path(dupire_identify.__file__).resolve().parent/"kernels"/"pricer.h"
    repair(installed)
    project=Path(__file__).resolve().parents[1]/"src"/"dupire_identify"/"kernels"/"pricer.h"
    if project.exists() and project.resolve()!=installed:repair(project)
    print("Repair complete. Run doctor --backend cuda in a new process.")
