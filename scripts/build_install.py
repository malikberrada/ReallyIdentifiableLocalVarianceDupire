"""Build a NEW wheel from this patch and install it in the running Python.

Windows example (existing venv):
    .venv/Scripts/python.exe scripts/build_install.py --backend cuda
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def run(arguments, *, cwd, env=None):
    print("\n> "+subprocess.list2cmdline([str(x) for x in arguments]),flush=True)
    subprocess.run([str(x) for x in arguments],cwd=cwd,env=env,check=True)


def main():
    parser=argparse.ArgumentParser(description="Build the source patch, install its wheel, run doctor")
    parser.add_argument("--backend",choices=("cuda","auto","numpy","cpp"),default="cuda")
    parser.add_argument("--native",action="store_true",help="Also compile C++/OpenMP; requires a C++ compiler")
    parser.add_argument("--skip-doctor",action="store_true")
    parser.add_argument("--no-build-isolation",action="store_true",
                        help="Use the already installed setuptools/wheel/pybind11 (offline environments)")
    parser.add_argument("--dist",type=Path,help="Output directory (default: this project's dist)")
    args=parser.parse_args()
    if args.backend=="cpp" and not args.native:
        parser.error("--backend cpp requires --native")
    project=Path(__file__).resolve().parents[1]
    destination=(args.dist or project/"dist").resolve()
    build_env=os.environ.copy()
    build_env["DUPIRE_NATIVE"]="1" if args.native else "0"
    print(f"Python: {sys.executable}\nSource: {project}\nOutput: {destination}",flush=True)
    # Build from a clean snapshot: an old build/lib or a compiled .pyd/.so
    # must not leak into a newly generated portable wheel.
    ignore=shutil.ignore_patterns("__pycache__","*.pyc","*.pyd","*.so","*.so.*",
                                  "*.egg-info",".pytest_cache",".venv","build","dist")
    with tempfile.TemporaryDirectory(prefix="dupire-wheel-") as temporary:
        stage=Path(temporary)/"source"
        stage.mkdir()
        print("\n[1/4] Preparing clean source",flush=True)
        for name in ("pyproject.toml","setup.py","MANIFEST.in","README.md","LICENSE"):
            shutil.copy2(project/name,stage/name)
        for name in ("src","docs","scripts","tests","examples",".github"):
            if (project/name).is_dir():shutil.copytree(project/name,stage/name,ignore=ignore)
        output=Path(temporary)/"wheels"
        print("\n[2/4] Building wheel from source",flush=True)
        build_args=[sys.executable,"-m","pip","wheel","--no-deps","--no-cache-dir"]
        if args.no_build_isolation:build_args.append("--no-build-isolation")
        run(build_args+["--wheel-dir",output,stage],cwd=stage,env=build_env)
        wheels=list(output.glob("dupire_identify-*.whl"))
        if len(wheels)!=1:raise RuntimeError(f"Expected one newly built wheel; found {len(wheels)}")
        if not args.native:
            with zipfile.ZipFile(wheels[0]) as archive:
                if any(n.endswith((".pyd",".so",".dll")) for n in archive.namelist()):
                    raise RuntimeError("Unexpected native binary in portable wheel")
        destination.mkdir(parents=True,exist_ok=True)
        wheel=destination/wheels[0].name
        pending=wheel.with_suffix(".whl.tmp")
        shutil.copy2(wheels[0],pending)
        pending.replace(wheel)
        checksums=[]
        for artifact in sorted(destination.iterdir()):
            if not artifact.is_file() or not (artifact.name.endswith('.whl') or artifact.name.endswith('.tar.gz')):continue
            with artifact.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            checksums.append(digest+'  '+artifact.name)
        checksum_pending=destination/'SHA256SUMS.txt.tmp'
        checksum_pending.write_text('\n'.join(checksums)+'\n',encoding='utf-8')
        checksum_pending.replace(destination/'SHA256SUMS.txt')
    print(f"\n[3/4] Installing generated wheel: {wheel}",flush=True)
    # Resolve newly introduced requirements (e.g. highspy), keeping installed
    # compatible NumPy/SciPy/CuPy versions. Then force this exact source wheel.
    run([sys.executable,"-m","pip","install",wheel],cwd=project)
    run([sys.executable,"-m","pip","install","--force-reinstall","--no-deps",wheel],cwd=project)
    print("\n[4/4] "+("Doctor skipped" if args.skip_doctor else "Checking installed package"),flush=True)
    if not args.skip_doctor:
        run([sys.executable,"-m","dupire_identify","doctor","--backend",args.backend],cwd=project)
    print(f"\nDONE\nWheel: {wheel}\nInstalled with: {sys.executable}",flush=True)


if __name__=="__main__":
    try:main()
    except subprocess.CalledProcessError as error:
        print(f"\nFAILED: command exited with status {error.returncode}",file=sys.stderr,flush=True)
        raise SystemExit(error.returncode or 1)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"\nFAILED: {error}",file=sys.stderr,flush=True)
        raise SystemExit(1)
