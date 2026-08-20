
import os
import json
import sys

CONFIG_FILE   = os.path.join("config", "taxa_list.json")
MANIFEST_FILE = os.path.join("results", "models", "split_manifest.json")
GENOMES_DIR   = os.path.join("data", "new_genomes")

# Organismos reservados para benchmark retrospectivo — NUNCA deben aparecer
# en config/taxa_list.json (ni por nombre ni por taxon_id real conocido).
RESERVED_FOR_BENCHMARK = {
    "Penicillium_chrysogenum": "5076",
    "Pseudoalteromonas_haloplanktis": "228",
    "Cryomyces_antarcticus": "329879",
}

errors, warnings_ = [], []


def check_taxa_list():
    print("── 1) config/taxa_list.json — organismos reservados para benchmark ──")
    if not os.path.exists(CONFIG_FILE):
        warnings_.append(f"No existe {CONFIG_FILE} — no se pudo chequear.")
        print(f"  ⚠️  No existe {CONFIG_FILE}")
        return

    with open(CONFIG_FILE) as f:
        taxa = json.load(f)

    all_entries = [(b, e) for b in ("psychrophiles", "mesophiles") for e in taxa.get(b, [])]
    all_ids   = {e["taxon_id"] for _, e in all_entries}
    all_names = {e["name"] for _, e in all_entries}

    for name, tid in RESERVED_FOR_BENCHMARK.items():
        if tid in all_ids:
            errors.append(f"taxon_id {tid} ({name}) está en taxa_list.json — "
                           f"contamina el benchmark retrospectivo.")
            print(f"  🚨 taxon_id {tid} ({name}) SIGUE en taxa_list.json")
        elif name in all_names:
            errors.append(f"nombre '{name}' está en taxa_list.json (id distinto) — revisar a mano.")
            print(f"  🚨 nombre '{name}' está en taxa_list.json — revisar (id distinto al esperado)")
        else:
            print(f"  ✅ {name} (id={tid}) ausente de taxa_list.json")

    # IDs duplicados dentro del propio archivo
    seen = {}
    for bucket, e in all_entries:
        tid = e["taxon_id"]
        if tid in seen:
            errors.append(f"taxon_id {tid} duplicado: '{seen[tid]}' y '{bucket}/{e['name']}'")
            print(f"  🚨 taxon_id {tid} duplicado: '{seen[tid]}' y '{bucket}/{e['name']}'")
        seen[tid] = f"{bucket}/{e['name']}"

    print(f"  Total organismos en taxa_list.json: {len(all_entries)}\n")


def check_manifest():
    print("── 2) split_manifest.json — disjunción train/test + umbral ──")
    if not os.path.exists(MANIFEST_FILE):
        warnings_.append(f"No existe {MANIFEST_FILE} — corre 05_train_model.py primero.")
        print(f"  ⚠️  No existe {MANIFEST_FILE} — corre 05_train_model.py primero.\n")
        return

    with open(MANIFEST_FILE) as f:
        m = json.load(f)

    train_orgs = set(m.get("train_organisms", []))
    test_orgs  = set(m.get("test_organisms", []))
    overlap_orgs = train_orgs & test_orgs
    if overlap_orgs:
        errors.append(f"{len(overlap_orgs)} organismo(s) en train Y test: {overlap_orgs}")
        print(f"  🚨 Organismos repetidos entre train/test: {overlap_orgs}")
    else:
        print(f"  ✅ train ({len(train_orgs)} organismos) y test ({len(test_orgs)}) son disjuntos")

    train_ids = set(m.get("train_protein_ids", []))
    test_ids  = set(m.get("test_protein_ids", []))
    overlap_ids = train_ids & test_ids
    if overlap_ids:
        errors.append(f"{len(overlap_ids)} Protein_ID en train Y test.")
        print(f"  🚨 {len(overlap_ids)} Protein_ID repetidos entre train/test")
    else:
        print(f"  ✅ {len(train_ids):,} proteínas train / {len(test_ids):,} test — sin overlap de ID")

    method = m.get("threshold_method")
    if method == "oof_cv_train_only":
        print(f"  ✅ Umbral ({m.get('threshold'):.4f}) fijado vía CV en train (sin leakage)")
    else:
        warnings_.append("El manifiesto no indica threshold_method=oof_cv_train_only — "
                          "¿se usó 05_train_model.py parcheado?")
        print(f"  ⚠️  El manifiesto no trae 'threshold_method' — probablemente se generó "
              f"con la versión SIN el fix de umbral. Vuelve a correr 05_train_model.py.")

    # Cruce contra los organismos reservados para benchmark
    reserved_names = set(RESERVED_FOR_BENCHMARK.keys())
    hit = reserved_names & (train_orgs | test_orgs)
    if hit:
        errors.append(f"Organismos reservados para benchmark aparecen en el manifest: {hit}")
        print(f"  🚨 Organismos reservados para benchmark en train/test del modelo: {hit}")
    print()


def check_new_genomes():
    print("── 3) data/new_genomes — archivos listos para benchmark externo ──")
    if not os.path.isdir(GENOMES_DIR):
        print(f"  ⚠️  No existe {GENOMES_DIR}/ todavía.\n")
        return
    fastas = [f for f in os.listdir(GENOMES_DIR) if f.endswith(".fasta")]
    if not fastas:
        print(f"  ⚠️  Sin .fasta en {GENOMES_DIR}/ — coloca ahí los genomas de benchmark "
              f"(ej. Pseudoalteromonas_haloplanktis_TAC125.fasta, "
              f"Penicillium_chrysogenum.fasta) antes de correr 09/benchmark_known_enzymes.py.\n")
        return
    for f in fastas:
        print(f"  · {f}")
    print()


def main():
    print("\n" + "=" * 68)
    print("  PsychroScan — Sanity check anti-leakage (v1)")
    print("=" * 68 + "\n")

    check_taxa_list()
    check_manifest()
    check_new_genomes()

    print("=" * 68)
    if errors:
        print(f"  🚨 {len(errors)} PROBLEMA(S) CRÍTICO(S) — no publiques resultados todavía:")
        for e in errors:
            print(f"     · {e}")
    else:
        print("  ✅ Sin problemas críticos detectados.")
    if warnings_:
        print(f"\n  ⚠️  {len(warnings_)} advertencia(s):")
        for w in warnings_:
            print(f"     · {w}")
    print("=" * 68 + "\n")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()