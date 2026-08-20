

import json
import os
import re
import sys
import warnings

OX_RE = re.compile(r'OX=(\d+)')
OS_RE = re.compile(r'OS=(.*?)(?:\s+OX=|\s+GN=|\s+PE=|\s+SV=|$)')


def parse_taxon_id(header: str) -> str | None:
    """Extrae el taxon ID (campo OX=) de un header FASTA de UniProt."""
    m = OX_RE.search(header)
    return m.group(1) if m else None


def parse_organism_name(header: str) -> str | None:
    """Extrae el nombre de organismo (campo OS=) de un header FASTA de UniProt."""
    m = OS_RE.search(header)
    return m.group(1).strip() if m else None


def load_taxa_config(config_path: str) -> dict:
    """
    Carga config/taxa_list.json (privado) o config/taxa_list_example.json
    (fallback público) y devuelve un índice por taxon_id:

        { "314265": {"name": "Colwellia_psychrerythraea",
                      "thermal_class": 0, "t_opt_c": 4,
                      "source": "Arctic marine sediment"}, ... }

    thermal_class: 0 = psychrophile/psychrotroph, 1 = mesophile.
    """
    if not os.path.exists(config_path):
        example = os.path.join(os.path.dirname(config_path), "taxa_list_example.json")
        if os.path.exists(example):
            warnings.warn(
                f"{config_path} no encontrado. Usando {example} (dataset de ejemplo, "
                f"no el curado real) — los resultados NO serán representativos."
            )
            config_path = example
        else:
            print(f"❌ No se encontró {config_path} ni un archivo de ejemplo.")
            sys.exit(1)

    with open(config_path) as f:
        raw = json.load(f)

    index = {}
    for entry in raw.get("psychrophiles", []):
        index[str(entry["taxon_id"])] = {
            "name": entry["name"], "thermal_class": 0,
            "t_opt_c": entry.get("T_opt_C"), "source": entry.get("source", ""),
        }
    for entry in raw.get("mesophiles", []):
        index[str(entry["taxon_id"])] = {
            "name": entry["name"], "thermal_class": 1,
            "t_opt_c": entry.get("T_opt_C"), "source": entry.get("source", ""),
        }
    return index


def resolve_organism(header: str, taxa_index: dict, expected_thermal_class: int,
                      fallback_label: str) -> dict:
    """
    Resuelve el organismo real de una secuencia y valida contra taxa_list.

    Devuelve dict con:
      - Organism_Source : nombre real del organismo (o fallback si no se pudo resolver)
      - Taxon_ID        : taxon ID real (o None)
      - T_opt_C         : temperatura óptima documentada (o None)
      - Organism_Resolved: True/False — si se pudo cruzar con taxa_list.json
      - Label_Mismatch  : True si el thermal_class del archivo (Cold_/Warm_)
                          no coincide con el thermal_class registrado en taxa_list
                          para ese organismo (señal de error de curación).
    """
    taxon_id = parse_taxon_id(header)
    entry = taxa_index.get(taxon_id) if taxon_id else None

    if entry is not None:
        mismatch = entry["thermal_class"] != expected_thermal_class
        return {
            "Organism_Source":   entry["name"],
            "Taxon_ID":          taxon_id,
            "T_opt_C":           entry["t_opt_c"],
            "Organism_Resolved": True,
            "Label_Mismatch":    mismatch,
        }

    # Fallback: no se pudo cruzar con taxa_list (taxon_id ausente del header
    # o no está en la lista curada). Usamos el nombre OS= tal cual si existe.
    os_name = parse_organism_name(header)
    return {
        "Organism_Source":   os_name if os_name else f"Unknown_{fallback_label}",
        "Taxon_ID":          taxon_id,
        "T_opt_C":           None,
        "Organism_Resolved": False,
        "Label_Mismatch":    False,
    }
