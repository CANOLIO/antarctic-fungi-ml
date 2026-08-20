
import os
import re
import json
import time
import requests

CONFIG_FILE  = os.path.join("config", "taxa_list.json")
OUT_FILE     = os.path.join("config", "taxa_list_rebuilt.json")
REPORT_FILE  = os.path.join("results", "taxa_list_rebuild_report.json")
SEARCH_API   = "https://rest.uniprot.org/taxonomy/search"

ACCEPTABLE_RANKS = {"species", "subspecies", "strain", "no rank", "varietas", "forma"}


def name_to_query(name: str) -> str:
    """Convierte 'Rhodococcus_erythropolis' -> 'Rhodococcus erythropolis'."""
    # Quita sufijos de cepa tipo _34H, _ATCC7966, _K12 para buscar la especie base;
    # se guarda el sufijo aparte para preferir, si existe, un match de esa cepa.
    clean = name.replace("_", " ").strip()
    return clean


def search_taxon_by_name(query: str) -> list:
    try:
        resp = requests.get(SEARCH_API, params={"query": query, "format": "json", "size": 10}, timeout=15)
        if resp.status_code != 200:
            return []
        return resp.json().get("results", [])
    except Exception:
        return []


def pick_best_match(query: str, results: list) -> dict | None:
    """
    Prioriza: (1) match exacto de scientificName con rank=species,
    (2) match exacto de scientificName con cualquier rank aceptable,
    (3) si no hay exacto, ninguno (queda para revisión manual).
    """
    query_norm = query.lower().strip()
    exact_species = [r for r in results
                     if r.get("scientificName", "").lower() == query_norm
                     and r.get("rank", "").lower() == "species"]
    if exact_species:
        return exact_species[0]

    exact_any = [r for r in results
                 if r.get("scientificName", "").lower() == query_norm
                 and r.get("rank", "").lower() in ACCEPTABLE_RANKS]
    if exact_any:
        return exact_any[0]

    return None


def rebuild():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ No se encontró {CONFIG_FILE}")
        return

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    report = {"corrected": [], "unchanged_ok": [], "needs_manual_review": []}
    new_config = {"psychrophiles": [], "mesophiles": []}

    print("\n" + "=" * 78)
    print("  Reconstruyendo taxa_list.json — buscando taxon_id correcto por NOMBRE")
    print("=" * 78 + "\n")

    for bucket in ("psychrophiles", "mesophiles"):
        for entry in config.get(bucket, []):
            name, old_id = entry["name"], entry["taxon_id"]
            query = name_to_query(name)
            results = search_taxon_by_name(query)
            time.sleep(0.34)

            best = pick_best_match(query, results)

            if best is None:
                # Reintentar sin el posible sufijo de cepa (ej. "_34H", "_K12", "_ATCC7966")
                base_query = re.sub(r'\s+(K12|ATCC\w*|DSM\w*|[A-Z0-9]{2,}[- ]?\d*)$', '', query,
                                    flags=re.IGNORECASE).strip()
                if base_query != query:
                    results2 = search_taxon_by_name(base_query)
                    time.sleep(0.34)
                    best = pick_best_match(base_query, results2)
                    if best:
                        query = base_query

            if best is None:
                print(f"  ❓ {name:<38} — sin match exacto en UniProt, requiere revisión manual")
                report["needs_manual_review"].append({
                    "name": name, "old_taxon_id": old_id, "bucket": bucket,
                    "query_tried": query,
                    "candidates": [{"id": r.get("taxonId"), "name": r.get("scientificName"),
                                   "rank": r.get("rank")} for r in results[:5]],
                })
                new_config[bucket].append(entry)  # se mantiene el original, sin tocar
                continue

            new_id = best.get("taxonId")
            if str(new_id) == str(old_id):
                print(f"  ✅ {name:<38} (id={old_id}) — ya estaba correcto")
                report["unchanged_ok"].append({"name": name, "taxon_id": old_id, "bucket": bucket})
            else:
                print(f"  🔧 {name:<38} id {old_id:>8} → {new_id:>8}  "
                      f"({best.get('scientificName')}, rank={best.get('rank')})")
                report["corrected"].append({
                    "name": name, "bucket": bucket,
                    "old_taxon_id": old_id, "new_taxon_id": str(new_id),
                    "resolved_name": best.get("scientificName"), "rank": best.get("rank"),
                })

            new_config[bucket].append({"taxon_id": str(new_id), "name": name})

    os.makedirs("results", exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("  RESUMEN")
    print("=" * 78)
    print(f"  Ya estaban correctos      : {len(report['unchanged_ok'])}")
    print(f"  Corregidos automáticamente: {len(report['corrected'])}")
    print(f"  Requieren revisión manual : {len(report['needs_manual_review'])}")
    print(f"\n  Nuevo archivo (NO reemplaza el original) → {OUT_FILE}")
    print(f"  Reporte detallado                        → {REPORT_FILE}")
    if report["needs_manual_review"]:
        print(f"\n  ⚠️  Revisa manualmente los {len(report['needs_manual_review'])} casos sin match")
        print(f"     exacto (probablemente sinónimos o nombres desactualizados) antes de")
        print(f"     reemplazar config/taxa_list.json.")
    print(f"\n  Una vez que confirmes {OUT_FILE}, renómbralo a taxa_list.json,")
    print(f"  borra data/raw/industrial_enzymes/*.fasta y _descarga_completada.log,")
    print(f"  y volvé a correr 01b → 03 → 05 desde cero.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    rebuild()
