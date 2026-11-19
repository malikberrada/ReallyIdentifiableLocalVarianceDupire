"""Certifier une estimation de la moyenne pondérée de variance locale."""
import argparse
from fractions import Fraction
from hashlib import file_digest, sha256
import json
from math import ceil
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import BadZipFile, ZipFile


REPO = Path(__file__).resolve().parents[1]
R3_SHA256 = "1d216a6401d826b35c690f2d26fba852be0720ba2497e41ed9aed7cd0ba1625d"
SCHEMA = "dupire_identify.garantie_positive.v1"


def decimal_units(units, digits):
    whole, remainder = divmod(abs(units), 10**digits)
    sign = "-" if units < 0 else ""
    return f"{sign}{whole}.{remainder:0{digits}d}" if digits else f"{sign}{whole}"


def estimate(lower, upper, epsilon, digits):
    """Bounds for the exact midpoint and for the decimal value the user can apply."""
    if lower > upper:
        raise ValueError("Bornes extérieures contradictoires.")
    midpoint = (lower + upper) / 2
    radius = (upper - lower) / 2
    scale = 10**digits
    units = round(midpoint * scale)
    displayed = Fraction(units, scale)
    display_radius = max(abs(displayed - lower), abs(upper - displayed))
    # Ceiling in integer arithmetic, including an exactly representable radius.
    error_units = ceil(display_radius * scale)
    return {
        "intervalle_exterieur": [str(lower), str(upper)],
        "estimation_exacte": str(midpoint),
        "erreur_absolue_au_plus": str(radius),
        "precision_cible_garantie": radius <= epsilon,
        "decimal": {
            "estimation": decimal_units(units, digits),
            "erreur_absolue_au_plus": decimal_units(error_units, digits),
            "precision_cible_garantie": Fraction(error_units, scale) <= epsilon,
        },
    }


def verified_leaves(node, lower, upper):
    """A parent's verified outer bounds also apply on all its children."""
    if "lower" in node:
        lower = max(lower, Fraction(node["lower"]))
        upper = min(upper, Fraction(node["upper"]))
    if lower > upper:
        raise ValueError("Intersection vide des bornes vérifiées.")
    if node["status"] == "split":
        for child in node["children"]:
            yield from verified_leaves(child, lower, upper)
    else:
        yield node["path"], node["family"], lower, upper


def general(run):
    sys.path.insert(0, str(REPO / "src"))
    from dupire_identify import general_frontier

    manifest_path = run / "general_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    hashes = json.loads(manifest_bytes)["artifacts"]
    frontier_bytes = (run / "frontier.json").read_bytes()
    if sha256(frontier_bytes).hexdigest() != hashes["frontier.json"]:
        raise ValueError("La frontière diffère de son manifeste.")
    verification = general_frontier.verify(run)
    if not verification["verified"] or verification["model_class_empty_certified"]:
        raise ValueError("Le domaine doit admettre au moins un modèle compatible.")
    if manifest_path.read_bytes() != manifest_bytes or (run / "frontier.json").read_bytes() != frontier_bytes:
        raise ValueError("La frontière a changé pendant la vérification.")
    data = json.loads(frontier_bytes)
    if not data["model_class_nonempty_certified"] or data["verified_model_count"] < 1:
        raise ValueError("Aucun modèle compatible vérifié pour ces cotations.")
    settings = data["settings"]
    low, high = map(Fraction, (settings["amin"], settings["amax"]))
    if not 0 < low <= high:
        raise ValueError("Les bornes de variance doivent être strictement positives et ordonnées.")
    leaves = list(verified_leaves(data["tree"], low, high))
    source = {
        "methode": "verify-general",
        "cotations_sha256": hashes["normalized.csv"],
        "nombre_cotations": len(settings["quotes"]),
        "variance_a_priori": [str(low), str(high)],
        "frontiere_sha256": hashes["frontier.json"],
        "modeles_compatibles_verifies": data["verified_model_count"],
    }
    return data["tree"]["family"], leaves, source


def from_quotes(args):
    sys.path.insert(0, str(REPO / "src"))
    from dupire_identify.families import Family
    from dupire_identify.io import read_quotes
    from dupire_identify.positive_quotes import bounds, gamma_bounds

    with args.quotes.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    quotes = read_quotes(args.quotes)
    if args.mode == "gamma":
        kernel, lo, hi, evidence = gamma_bounds(quotes, args.call_id, args.variance_reference,
                                               args.amin, args.amax, args.variance_temoin, args.modele_temoin,
                                               args.corridor)
        method = "identite_de_couverture_du_call"
    else:
        family = Family.make(json.loads(args.domaine.read_bytes()))
        lo, hi, evidence = bounds(quotes, family, args.amin, args.amax, args.variance_temoin,
                                  args.modele_temoin, args.cellules)
        kernel, method = family.json(), "budgets_gamma_depuis_les_cotations"
    with args.quotes.open("rb") as stream:
        if digest != file_digest(stream, "sha256").hexdigest():
            raise ValueError("Le CSV a changé pendant le calcul.")
    source = {
        "methode": method,
        "cotations_sha256": digest,
        "nombre_cotations": len(quotes),
        "variance_a_priori": [str(args.amin), str(args.amax)],
        **evidence,
    }
    return kernel, [("r", kernel, lo, hi)], source


def r3(run, workers):
    """Rebuild the calibrated HJB upper bound for the whole early R3 family."""
    archive = REPO / "rejeu_r3" / "campagne.zip"
    receipt_path = run / "simulation" / "proof-replay.json"
    with archive.open("rb") as stream, TemporaryDirectory(prefix="dupire-gp-") as tmp:
        if file_digest(stream, "sha256").hexdigest() != R3_SHA256:
            raise ValueError("Empreinte de la campagne R3 incorrecte.")
        stream.seek(0)
        with ZipFile(stream) as bundle:
            receipt_bytes = receipt_path.read_bytes()
            if receipt_bytes != bundle.read("resultats/proof-replay.json"):
                raise ValueError("Le reçu du rejeu ne correspond pas à cette campagne R3.")
            receipt = json.loads(receipt_bytes)
            local_proofs = run / "optimisation" / "proofs.json"
            proofs = local_proofs.read_bytes() if local_proofs.exists() else bundle.read("proofs.json")
            if not receipt["verified"] or sha256(proofs).hexdigest() != receipt["proofs_sha256"]:
                raise ValueError("Registre de preuves R3 différent du registre vérifié.")
            records = [p for p in json.loads(proofs) if p["kind"] == "hjb" and p.get("folder") == "hjb-early"]
            if len(records) != 1 or records[0]["endpoint"] != "maximum":
                raise ValueError("Certificat HJB de la famille précoce introuvable ou ambigu.")
            record = records[0]
            for name in bundle.namelist():
                if name.endswith(".py") or name.startswith("hjb-early/") or name == "normalized.csv":
                    bundle.extract(name, tmp)

        # The original R3 engine and the 0.5.1 source use the same package name.
        # R3 is a separate CLI invocation; load the authenticated campaign first.
        campaign = Path(tmp)
        sys.path[:0] = [str(campaign), str(campaign / "base" / "src"), str(campaign / "base")]
        from flows import CAP, FLOOR, QUOTE_HASH, REF, call, original_quotes
        from hjb_replay import upper_bound
        from dupire_identify.denominator import brownian_floor
        from dupire_identify.families import Family

        quotes = original_quotes()
        for quote in quotes:
            price = call(quote.k, REF * quote.t)
            if not quote.bid <= price.lo <= price.hi <= quote.ask:
                raise ValueError(f"Modèle de référence hors spread pour {quote.id}.")
        family = Family.make({"coordinates": "left", **dict(zip(("u", "v", "ht", "hk"), record["family"], strict=True))})
        shared = family.common_kernel()
        if shared is None:
            raise ValueError("Noyau commun de la famille R3 manquant.")
        window, factor = shared
        floor = brownian_floor(window, FLOOR, CAP)
        if not floor or factor * Fraction(floor["lower_bound"]) <= 0:
            raise ValueError("Positivité du dénominateur non certifiée.")
        denominator_lower = factor * Fraction(floor["lower_bound"])
        upper, denominator_upper = upper_bound(record, workers, rebuild=True)
        if upper != Fraction(record["upper"]) or denominator_upper != Fraction(record["denominator_upper"]):
            raise ValueError("Écart lors du recalcul du majorant HJB.")
        if not FLOOR <= REF <= upper <= CAP:
            raise ValueError("Borne HJB incompatible avec le modèle de référence ou le prior.")
        source = {
            "methode": "recalcul_hjb_early",
            "campagne_sha256": R3_SHA256,
            "registre_sha256": receipt["proofs_sha256"],
            "cotations_sha256": QUOTE_HASH,
            "nombre_cotations": len(quotes),
            "variance_a_priori": [str(FLOOR), str(CAP)],
            "modele_compatible_variance_constante": str(REF),
            "minorant_denominateur_noyau_original": str(denominator_lower),
            "certificat_hjb_sha256": record["files"]["static_start_certificate.json"],
        }
        return family.json(), [("hjb-early", family.json(), FLOOR, upper)], source


def report_for(family, leaves, source, epsilon, digits):
    # One constant estimator must cover every real window in the requested box.
    lo = min(row[2] for row in leaves)
    hi = max(row[3] for row in leaves)
    prior_lo, prior_hi = map(Fraction, source["variance_a_priori"])
    result = {
        "schema": SCHEMA,
        "observable": "E[int phi a S^2 dt] / E[int phi S^2 dt]",
        "unite": "variance locale annuelle",
        "source": source,
        "famille": family,
        "epsilon_demande": str(epsilon),
        "intervalle_resserre_par_rapport_au_prior": lo > prior_lo or hi < prior_hi,
        "garantie_uniforme": estimate(lo, hi, epsilon, digits),
    }
    if len(leaves) > 1:
        result["garanties_par_feuille"] = [
            {"chemin": name, "famille": box, **estimate(a, b, epsilon, digits)}
            for name, box, a, b in leaves
        ]
    return result


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_bytes())
        if not isinstance(previous, dict) or previous.get("schema") != SCHEMA:
            raise ValueError(f"Choisir un autre fichier de sortie pour conserver {path}.")
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    gen = sub.add_parser("general", help="Famille d'un calcul solve-general")
    gen.add_argument("--run", type=Path, default=REPO / "examples" / "general_informative")
    market = sub.add_parser("cotations", help="Construire un certificat depuis un CSV normalisé et une famille")
    market.add_argument("--domaine", type=Path, required=True, help="JSON Family du projet")
    market.add_argument("--cellules", type=int, default=16)
    gamma = sub.add_parser("gamma", help="Moyenne pondérée par le gamma Black d'un call coté")
    gamma.add_argument("--call-id", required=True, help="Identifiant du call dans le CSV")
    gamma.add_argument("--variance-reference", type=Fraction, required=True, help="Variance Black définissant le noyau")
    gamma.add_argument("--corridor", nargs=2, type=Fraction, metavar=("BAS", "HAUT"),
                       help="Restreindre le poids gamma aux prix normalisés dans ce corridor")
    for cmd in (market, gamma):
        cmd.add_argument("--quotes", type=Path, required=True)
        model = cmd.add_mutually_exclusive_group()
        model.add_argument("--variance-temoin", type=Fraction, help="Variance constante à vérifier sur toutes les cotations")
        model.add_argument("--modele-temoin", type=Path, help="Dossier d'un modèle compatible vérifiable")
        cmd.add_argument("--amin", type=Fraction, default=Fraction(1, 400))
        cmd.add_argument("--amax", type=Fraction, default=Fraction(1, 4))
    early = sub.add_parser("r3", help="Famille précoce du rejeu R3")
    early.add_argument("--run", type=Path, required=True, help="Dossier contenant simulation/proof-replay.json")
    early.add_argument("--workers", type=int, choices=range(1, 17), default=4, metavar="1..16")
    for cmd in (gen, market, gamma, early):
        cmd.add_argument("--epsilon", type=Fraction, required=True, help="Erreur absolue cible, p. ex. 1/100 ou 0.0526")
        cmd.add_argument("--decimales", type=int, default=10, help="Décimales de l'estimation affichée")
        cmd.add_argument("--out", type=Path, required=True, help="Fichier JSON du certificat")
    args = parser.parse_args()
    if args.epsilon < 0 or args.decimales < 0:
        parser.error("epsilon et decimales doivent être positifs ou nuls.")
    try:
        if args.mode in ("cotations", "gamma"):
            family, leaves, source = from_quotes(args)
        elif args.mode == "r3":
            family, leaves, source = r3(args.run.resolve(), args.workers)
        else:
            family, leaves, source = general(args.run.resolve())
        report = report_for(family, leaves, source, args.epsilon, args.decimales)
        write_report(args.out, report)
        bound = report["garantie_uniforme"]
        shown = bound["decimal"]
        print(f"Estimation décimale {shown['estimation']}  |  erreur absolue au plus {shown['erreur_absolue_au_plus']}")
        print(f"Précision demandée garantie (estimateur exact) = {bound['precision_cible_garantie']}")
        print(f"Précision demandée garantie (valeurs décimales affichées) = {shown['precision_cible_garantie']}")
        print(args.out.resolve())
        return 0 if bound["precision_cible_garantie"] else 2
    except (ArithmeticError, BadZipFile, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Vérification interrompue. {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
