"""Calculs R3 lancés dans un interpréteur isolé par scripts/rejouer_r3.py."""
from fractions import Fraction
from hashlib import file_digest
import json
import multiprocessing
from pathlib import Path
import shutil
import sys

RECIPES_SHA = "93c50f9e281cca1038d7b48948a93dcfa40e39a1ef2b23ea033e28445067a894"
COEFFICIENTS_SHA = "28d2327a2a8238f09ae45ce373e0c212c636d4f218520b392f1b2bab6d20fb16"
HJB_CASES = ("hjb", "hjb-early", "hjb-mixed")
POINTS = [
    ("11/20", "3/5", "29/50", "7/10"),
    ("1/4", "1/4", "1/4", "3/4"),
    ("57/100", "1/4", "3/10", "17/8"),
    ("14/25", "63/100", "77/100", "57/100"),
]
BUNDLE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj, *, compact=False):
    formatting = {"separators": (",", ":")} if compact else {"indent": 2}
    text = json.dumps(obj, ensure_ascii=False, allow_nan=False, **formatting)
    path.write_bytes((text + "\n").encode("utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def same(actual, reference):
    result = digest(actual)
    if result != digest(reference):
        raise ValueError(f"Écart par rapport à la référence pour {actual.name} (SHA-256 {result}).")
    return result


def solve_recorded_lp(case, grid, bank):
    """Solve the recorded Dinkelbach subproblems. The archive stores LP inputs only."""
    import warnings
    import numpy as np
    from scipy.optimize import OptimizeWarning, linprog
    from scipy.sparse import csr_matrix

    key = "g" + str(case["n"])
    constraints = csr_matrix((bank[key + "_data"], bank[key + "_indices"], bank[key + "_indptr"]),
                             shape=tuple(bank[key + "_shape"]))
    identity = case["model"]
    costs = bank["c_" + identity]
    methods = bank["method_" + identity]
    if not len(costs) or any(method != "highs-ds" for method in methods):
        raise ValueError("Sous-problèmes de flux LP inconnus.")
    for objective, method in zip(costs, methods, strict=True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning, message="Unrecognized options.*")
            result = linprog(objective, A_ub=constraints, b_ub=bank[key + "_b"],
                             A_eq=grid.Aeq, b_eq=grid.beq, bounds=(0, None), method=str(method),
                             options={"threads": 1, "primal_feasibility_tolerance": 1e-9,
                                      "dual_feasibility_tolerance": 1e-9})
        if not result.success:
            raise ValueError(f"Le solveur n’a pas reconstitué le flux {identity}. {result.message}")
    # Same conservation and integer rounding as the campaign's FlowSearch.propose.
    units = 2**44
    incoming = {0: units}
    layers = []
    offset = 0
    for step in range(1, case["n"]):
        arcs = []
        following = {}
        for node, mass in sorted(incoming.items()):
            i = offset + 2 * node
            low, high = np.maximum(0, result.x[i:i + 2])
            high_units = round(mass * high / (low + high)) if low + high > 0 else 0
            for jump, count in ((0, mass - high_units), (1, high_units)):
                if count:
                    arcs.append([node, jump, count])
                    following[node + jump] = following.get(node + jump, 0) + count
        layers.append(arcs)
        incoming = following
        offset += 2 * step
    return {"n": case["n"], "mass_units": units, "endpoint": case["endpoint"], "layers": layers}


def optimise_clocks(campaign, out):
    import numpy as np
    from flows import certify_prices
    from search_flows import FlowSearch
    recipe_path = BUNDLE / "horloges.json"
    if digest(recipe_path) != RECIPES_SHA:
        raise ValueError("Les objectifs LP figés se rapportent aux fenêtres de la campagne R3.")
    plans = read(recipe_path)
    grids = {}
    recovered = {}
    bank_path = BUNDLE / "coefficients_flux.npz"
    if digest(bank_path) != COEFFICIENTS_SHA:
        raise ValueError("Empreinte incorrecte des coefficients LP de référence.")
    with np.load(bank_path, allow_pickle=False) as bank:
        for case in plans:
            n = case["n"]
            if n not in grids:
                grids[n] = FlowSearch(n)
            point = tuple(map(Fraction, case["window"]))
            grid = grids[n]
            model_id = case["model"]
            archived = read(campaign / "models" / (model_id + ".json"))
            try:
                flow, _ = grid.propose(point, case["endpoint"], optimize_ratio=case["optimize_ratio"])
            except (ArithmeticError, ValueError):
                flow = None
            if flow is None or flow["layers"] != archived["layers"]:
                print(f"Recalcul LP sur coefficients binary64 figés pour {model_id[:12]}.", flush=True)
                flow = solve_recorded_lp(case, grid, bank)
            recovered[model_id] = flow
    if len(recovered) != len(plans):
        raise ValueError("Témoin dupliqué dans la liste de rejeu.")
    (out / "models").mkdir()
    used = set()
    models = sorted((campaign / "models").glob("*.json"))
    for path in models:
        if path.stem in recovered:
            flow = recovered[path.stem]
            used.add(path.stem)
        else:
            flow = read(path)
        # Smallest reference weight compatible with all 27 bid/ask intervals.
        flow["reference_weight"], prices = certify_prices(flow, fit=True)
        if len(prices) != 27:
            raise ValueError(f"Calibration incomplète pour {path.name}.")
        target = out / "models" / path.name
        # The campaign originally used two serializers for model fingerprints.
        with path.open("rb") as source:
            spaced = source.read(6) == b'{"n": '
        separators = None if spaced else (",", ":")
        target.write_bytes(json.dumps(flow, separators=separators, allow_nan=False).encode("ascii"))
        same(target, path)
    if used != set(recovered) or len(models) != 77:
        raise ValueError("Banque de modèles différente de la campagne.")
    print(f"{len(used)} flux LP régénérés, 77 poids de calibration recalculés.", flush=True)
    return len(used)


def optimise_calendar(campaign, out):
    from calendar_portfolio import calendar_budget, propose
    reference = campaign / "calendar-proofs.json"
    rebuilt = {}
    # Use only the spatial windows and lattice sizes as LP inputs.
    for i, (key, item) in enumerate(read(reference).items(), 1):
        space = tuple(map(Fraction, item["space"]))
        hedge = propose(space, item["hedge"]["count"])
        budget = str(calendar_budget(space, hedge))
        result = {"space": item["space"], "hedge": hedge, "budget": budget}
        if result != item:
            raise ValueError(f"Portefeuille ou budget différent pour le calendrier {key}.")
        rebuilt[key] = result
        if i % 250 == 0:
            print(f"{i} portefeuilles calendaires recalculés.", flush=True)
    write(out / reference.name, rebuilt, compact=True)
    same(out / reference.name, reference)
    return len(rebuilt)


def optimise_hedges(campaign, out):
    import calendar_cap
    import calendar_floor
    import hjb_mixed
    proofs = read(campaign / "proofs.json")
    counts = {"calendar": 0, "calendar_cap": 0, "hjb": 0, "hjb_mixed": 0}
    for entry in proofs:
        kind = entry["kind"]
        if kind in ("calendar", "hjb"):
            entry["hedge"] = calendar_floor.propose(entry["family"])
        elif kind == "calendar_cap":
            upper, entry["dual"] = calendar_cap.certify(entry["family"])
            entry["upper"] = str(upper)
        else:
            continue
        counts[kind] += 1
    write(out / "proofs.json", proofs, compact=True)
    same(out / "proofs.json", campaign / "proofs.json")
    extra = read(campaign / "supplement.json")
    for entry in extra:
        if entry["kind"] == "hjb_mixed":
            curves = entry["curves"]
            knots = curves[0]["knots"]
            if any(curve["knots"] != knots for curve in curves):
                raise ValueError("Maillages hétérogènes dans le portefeuille HJB.")
            entry["curves"] = hjb_mixed.propose(entry["family"], knots)
            counts["hjb_mixed"] += 1
    write(out / "supplement.json", extra, compact=True)
    same(out / "supplement.json", campaign / "supplement.json")
    return counts


def optimise_hjb(campaign, out, workers):
    from dupire_identify.static_start_hjb import build
    for case in HJB_CASES:
        print(f"Recalcul des portefeuilles et du certificat HJB pour {case}.", flush=True)
        report = build(campaign / case, workers=workers, saved=None)
        target = out / case / "static_start_certificate.json"
        target.parent.mkdir()
        write(target, report)
        same(target, campaign / case / target.name)


def optimise(campaign, out, workers):
    # FlowSearch and calendar_floor select one HiGHS thread before other LPs run.
    count = optimise_clocks(campaign, out)
    hedges = optimise_hedges(campaign, out)
    calendars = optimise_calendar(campaign, out)
    optimise_hjb(campaign, out, workers)
    write(out / "optimisation.json", {
        "identique": True, "portefeuilles_calendaires": calendars,
        "registres_avec_couvertures_recalculees": hedges,
        "flux_lp_regeneres": count, "flux_repris": 77 - count,
        "poids_de_calibration_recalcules": 77, "certificats_hjb_regeneres": len(HJB_CASES),
    })
    print(f"Optimisation identique à la référence pour {calendars} portefeuilles calendaires.", flush=True)


def simulate(campaign, out, workers, optimisation):
    import verify
    from fenetre import query
    if optimisation is not None:
        inputs = ["calendar-proofs.json", "proofs.json", "supplement.json"]
        inputs.extend(p.relative_to(campaign) for p in sorted((campaign / "models").glob("*.json")))
        inputs.extend(Path(case) / "static_start_certificate.json" for case in HJB_CASES)
        for name in inputs:
            src, dest = optimisation / name, campaign / name
            same(src, dest)
            shutil.copyfile(src, dest)
        print("La simulation utilise les entrées issues de l’optimisation.", flush=True)
    else:
        print("La simulation utilise les décisions enregistrées de la campagne.", flush=True)
    # The published receipts hash their child JSON files. Emit LF also on Windows.
    verify.save = write
    status = verify.verify(out, workers)
    completion = read(out / "completion.json")
    expected_status = 0 if completion["request_completed"] else 2
    if status != expected_status:
        raise ValueError(f"Code scientifique inattendu : {status}.")
    write(out / "points.json", query(POINTS, out))
    hashes = {}
    for reference in sorted((campaign / "resultats").glob("*.json")):
        actual = out / reference.name
        if actual.name.startswith("open-"):
            # These leaves are written by the immutable historical geometry module.
            raw = actual.read_bytes()
            if b"\r\n" in raw:
                actual.write_bytes(raw.replace(b"\r\n", b"\n"))
        hashes[actual.name] = same(actual, reference)
    write(out / "reproduction.json", {
        "reproduction_identique": True, "request_completed": completion["request_completed"],
        "code_du_verificateur": status, "sha256": hashes,
    })
    print(f"{len(hashes)} fichiers identiques, coverage.json = {hashes['coverage.json']}", flush=True)


def main():
    multiprocessing.set_start_method("spawn")
    stage, directory, output, worker_count, *extra = sys.argv[1:]
    campaign, out = Path(directory), Path(output)
    sys.path[:0] = [str(campaign), str(campaign / "base" / "src"), str(campaign / "base")]
    if stage == "optimiser":
        optimise(campaign, out, int(worker_count))
    elif stage == "simuler":
        simulate(campaign, out, int(worker_count), Path(extra[0]) if extra else None)
    else:
        raise ValueError(f"Étape inconnue : {stage}.")


if __name__ == "__main__":
    main()
