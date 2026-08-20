"""
validate_taxa_list.py — PsychroScan
=====================================
Valida que cada taxon_id en config/taxa_list.json sea una ESPECIE (o
subespecie/strain), no un rango taxonómico más alto (género, familia, orden,
clase, phylum). UniProt's `taxonomy_id:` es jerárquico: si un ID corresponde
a un rango alto, la query de 01b_fetch_brenda_coldenzymes.py trae TODOS los
descendientes de ese clado, contaminando el dataset silenciosamente.

Caso real detectado: taxon_id "1760" etiquetado como "Rhodococcus_erythropolis"
en realidad es la clase Actinobacteria (rank: class) — esto metió miles de
secuencias de Streptomyces, Mycobacterium, Micromonospora, etc. en el bucket
"Cold" del dataset de entrenamiento.

Uso:
    python src/utils/validate_taxa_list.py
    python src/utils/validate_taxa_list.py --fix-suggestions   # busca el ID correcto

Requiere acceso a internet (UniProt REST API — rest.uniprot.org).
"""
import os
import sys
import json
import time
import re
import requests

CONFIG_FILE = os.path.join("config", "taxa_list.json")
TAXONOMY_API = "https://rest.uniprot.org/taxonomy/{}.json"
SEARCH_API   = "https://rest.uniprot.org/taxonomy/search"

# Ranks aceptables — cualquier cosa por encima de 'species' en la jerarquía
# estándar de NCBI/UniProt es sospechoso para este proyecto (donde cada
# entrada debe representar UN organismo puntual, no un clado).
ACCEPTABLE_RANKS = {"species", "subspecies", "strain", "no rank", "varietas", "forma"}
SUSPICIOUS_RANKS = {"genus", "family", "order", "class", "phylum", "kingdom",
                     "superkingdom", "suborder", "subclass", "superfamily",
                     "tribe", "subphylum"}


def get_taxon_info(taxon_id: str) -> dict:
    try:
        resp = requests.get(TAXONOMY_API.format(taxon_id), timeout=15)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}"}
        data = resp.json()
        return {
            "scientificName": data.get("scientificName", "?"),
            "rank":            data.get("rank", "?"),
            "commonName":      data.get("commonName", ""),
            "otherNames":      data.get("otherNames", []),
        }
    except Exception as e:
        return {"error": str(e)}


def validate():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ No se encontró {CONFIG_FILE}")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    all_entries = [(e, "psychrophiles") for e in config.get("psychrophiles", [])] + \
                  [(e, "mesophiles")    for e in config.get("mesophiles", [])]

    print("\n" + "=" * 78)
    print(f"  Validando {len(all_entries)} taxon_ids contra UniProt Taxonomy REST API")
    print("=" * 78 + "\n")

    problems, ok_count = [], 0
    seen_ids = {}

    for entry, bucket in all_entries:
        tid, name = entry["taxon_id"], entry["name"]
        info = get_taxon_info(tid)
        time.sleep(0.34)  # ~3 req/s, cortés con la API

        if "error" in info:
            print(f"  ⚠️  {name:<38} (id={tid:<8}) [{bucket}] — error consultando: {info['error']}")
            problems.append({"name": name, "taxon_id": tid, "bucket": bucket,
                             "issue": f"query_error: {info['error']}"})
            continue

        rank = info["rank"].lower()
        sci  = info["scientificName"]

        if rank in SUSPICIOUS_RANKS:
            print(f"  🚨 {name:<38} (id={tid:<8}) [{bucket}] "
                  f"→ '{sci}' es RANK={rank.upper()} (¡no es una especie!)")
            problems.append({"name": name, "taxon_id": tid, "bucket": bucket,
                             "issue": f"rank={rank} (esperado: species)",
                             "actual_taxon": sci})
        elif rank not in ACCEPTABLE_RANKS:
            print(f"  ❓ {name:<38} (id={tid:<8}) [{bucket}] "
                  f"→ '{sci}' rank inusual: {rank} — revisar manualmente")
            problems.append({"name": name, "taxon_id": tid, "bucket": bucket,
                             "issue": f"rank inusual: {rank}", "actual_taxon": sci})
        else:
            ok_count += 1
            # Detectar nombre no coincidente (posible ID pegado a la entrada equivocada).
            # Se ignora el sufijo "(strain ...)" de UniProt para no marcar falsos positivos
            # en cepas correctamente resueltas (ej. "E. coli (strain K12)" vs "Escherichia_coli_K12").
            name_norm = name.replace("_", " ").lower().strip()
            sci_base  = re.sub(r'\s*\(strain[^)]*\)', '', sci, flags=re.IGNORECASE).strip().lower()
            sci_clean = sci.replace("[", "").replace("]", "").strip().lower()
            
            # Revisar coincidencia en scientificName o en sinónimos conocidos (otherNames)
            matched = (
                name_norm in sci.lower() or sci.lower() in name_norm
                or name_norm in sci_base or sci_base in name_norm
                or name_norm in sci_clean or sci_clean in name_norm
            )
            if not matched and "otherNames" in info:
                for syn in info["otherNames"]:
                    syn_clean = syn.replace("[", "").replace("]", "").strip().lower()
                    if name_norm in syn_clean or syn_clean in name_norm:
                        matched = True
                        break

            if not matched:
                print(f"  ⚠️  {name:<38} (id={tid:<8}) [{bucket}] "
                      f"→ nombre no coincide con UniProt: '{sci}' (rank OK: {rank})")
                problems.append({"name": name, "taxon_id": tid, "bucket": bucket,
                                 "issue": "nombre no coincide con UniProt",
                                 "actual_taxon": sci})

        # Duplicados / anidamiento: mismo taxon_id usado dos veces
        if tid in seen_ids:
            print(f"  ⚠️  taxon_id {tid} usado más de una vez: "
                  f"'{seen_ids[tid]}' y '{name}'")
            problems.append({"name": name, "taxon_id": tid, "bucket": bucket,
                             "issue": f"taxon_id duplicado (también usado por {seen_ids[tid]})"})
        seen_ids[tid] = name

    print("\n" + "=" * 78)
    print("  RESUMEN")
    print("=" * 78)
    print(f"  OK                    : {ok_count}/{len(all_entries)}")
    print(f"  Con problemas         : {len(problems)}/{len(all_entries)}")

    critical = [p for p in problems if "rank=" in p.get("issue", "") and "esperado" in p.get("issue", "")]
    if critical:
        print(f"\n  🚨 CRÍTICO — {len(critical)} entrada(s) apuntan a un clado, no a una especie:")
        print(f"     Estas contaminan el dataset con TODOS sus descendientes taxonómicos.")
        print(f"     Hay que corregir el taxon_id y volver a correr 01b + 03 + 05.")
        for p in critical:
            print(f"     · {p['name']} (id={p['taxon_id']}, bucket={p['bucket']}) "
                  f"→ en realidad es: {p.get('actual_taxon', '?')}")

    if problems:
        out_path = os.path.join("results", "taxa_list_validation.json")
        os.makedirs("results", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
        print(f"\n  Detalle completo → {out_path}")
    else:
        print(f"\n  ✅ Todos los taxon_ids resuelven a especies/strains individuales.")

    print("=" * 78 + "\n")
    return problems


if __name__ == "__main__":
    validate()
