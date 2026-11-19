"""Rejouer la campagne R3 du 18 novembre 2026 avec ses données figées."""
from argparse import ArgumentParser
from hashlib import file_digest, sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

BUNDLE = Path(__file__).resolve().parents[1] / "rejeu_r3"
ARCHIVE = BUNDLE / "campagne.zip"
ARCHIVE_SHA = "1d216a6401d826b35c690f2d26fba852be0720ba2497e41ed9aed7cd0ba1625d"
STAGES = {"optimiser": "optimisation", "simuler": "simulation"}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def files(root):
    return {f.relative_to(root).as_posix(): digest(f)
            for f in sorted(root.rglob("*")) if f.is_file()}


def publish(staging, target):
    # A concurrent run can finish first. Only accept an identical complete tree.
    try:
        staging.rename(target)
    except OSError:
        if not target.is_dir():
            raise
        if files(target) != files(staging):
            raise ValueError(f"{target} contient un autre résultat. Choisir un nouveau --out.")


def run_stage(stage, campaign, output, workers):
    target = output / STAGES[stage]
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUTF8="1")
    print(f"{STAGES[stage].capitalize()} → {target}", flush=True)
    with TemporaryDirectory(prefix=".r3-", dir=output) as temporary:
        staging = Path(temporary) / "calcul"
        staging.mkdir()
        command = [sys.executable, "-I", "-X", "utf8", "-u", str(BUNDLE / "calcul.py"),
                   stage, str(campaign), str(staging), str(workers)]
        if stage == "simuler" and (output / "optimisation").is_dir():
            command.append(str(output / "optimisation"))
        subprocess.run(command, env=env, check=True)
        publish(staging, target)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["tout", *STAGES, "comparer"])
    parser.add_argument("--out", type=Path, default=Path("runs/r3_20261118"))
    parser.add_argument("--workers", type=int, default=4, help="Processus de calcul pour les certificats HJB et la simulation (1 à 16).")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers doit être compris entre 1 et 16.")
    destination = args.out.resolve()
    try:
        if digest(ARCHIVE) != ARCHIVE_SHA:
            raise ValueError("L’archive de la campagne ne correspond pas à son empreinte publiée.")
        with ZipFile(ARCHIVE) as archive:
            if args.action == "comparer":
                hashes = {}
                for name in sorted(archive.namelist()):
                    if name.startswith("resultats/") and name.endswith(".json"):
                        path = destination / "simulation" / Path(name).name
                        hashes[path.name] = digest(path)
                        if hashes[path.name] != sha256(archive.read(name)).hexdigest():
                            raise ValueError(f"Résultat différent de la référence pour {path}.")
                status = json.loads((destination / "simulation/completion.json").read_bytes())
                receipt = json.loads((destination / "simulation/reproduction.json").read_bytes())
                expected = {
                    "reproduction_identique": True, "request_completed": status["request_completed"],
                    "code_du_verificateur": 0 if status["request_completed"] else 2, "sha256": hashes,
                }
                if receipt != expected:
                    raise ValueError("Le reçu reproduction.json diffère des fichiers recalculés.")
                print(f"{len(hashes)} fichiers identiques. request_completed = {status['request_completed']}")
                return 0
            for requirement in (BUNDLE / "requirements.txt").read_text(encoding="utf-8").splitlines():
                package, expected = requirement.split("==")
                if version(package) != expected:
                    raise ValueError(f"Version requise pour ce rejeu : {package}=={expected}.")
            destination.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix="dupire-r3-") as temporary:
                campaign = Path(temporary)
                # The entire archive is authenticated above, including the original source.
                archive.extractall(campaign)
                stages = STAGES if args.action == "tout" else (args.action,)
                for stage in stages:
                    run_stage(stage, campaign, destination, args.workers)
        return 0
    except (OSError, ValueError, PackageNotFoundError, subprocess.CalledProcessError) as error:
        print(f"Échec du rejeu. {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
