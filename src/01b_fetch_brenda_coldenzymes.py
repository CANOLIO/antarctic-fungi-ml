"""
PsychroScan — 01b_fetch_brenda_coldenzymes.py  (v3 — public version)
======================================================================
Descarga secuencias de enzimas industriales desde UniProt REST API
usando paginación por cursor.

La lista de taxones se carga desde:

    config/taxa_list.json   ← NO incluido en el repositorio público

Formato esperado:
{
  "psychrophiles": [{"taxon_id": "12345", "name": "Organism_name"}, ...],
  "mesophiles":    [{"taxon_id": "67890", "name": "Organism_name"}, ...]
}

Para colaboraciones o acceso al pipeline completo: ver README.md → Contact.
"""

import os
import sys
import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─── RUTAS ────────────────────────────────────────────────────────────────────
CONFIG_FILE = os.path.join("config", "taxa_list.json")
RAW_DIR     = os.path.join("data", "raw", "industrial_enzymes")
LOG_FILE    = os.path.join(RAW_DIR, "_descarga_completada.log")
os.makedirs(RAW_DIR, exist_ok=True)

# ─── PARÁMETROS ───────────────────────────────────────────────────────────────
PAGE_SIZE    = 500
MAX_PER_FILE = 3000
SLEEP_PAGE   = 0.8
SLEEP_FILE   = 1.5

# ─── CLASES EC INDUSTRIALES ───────────────────────────────────────────────────
EC_CLASSES = [
    ("3.1.1.3",  "Lipases"),
    ("3.4.21.-", "Serine_Proteases"),
    ("3.4.24.-", "Metalloproteasas"),
    ("3.2.1.1",  "Alpha_Amylases"),
    ("3.2.1.4",  "Cellulases"),
]


# ─── CARGA DE TAXONES ─────────────────────────────────────────────────────────
def load_taxa():
    if not os.path.exists(CONFIG_FILE):
        print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  ERROR: config/taxa_list.json no encontrado                  ║
  ║                                                              ║
  ║  La lista de taxones de entrenamiento no está incluida       ║
  ║  en el repositorio público de PsychroScan.                   ║
  ║                                                              ║
  ║  Para colaboraciones o acceso completo al pipeline:          ║
  ║  ver README.md → sección Contact                             ║
  ╚══════════════════════════════════════════════════════════════╝
""")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    cold = config.get("psychrophiles", [])
    warm = config.get("mesophiles", [])

    if not cold or not warm:
        print("  ERROR: taxa_list.json debe contener claves "
              "'psychrophiles' y 'mesophiles'.")
        sys.exit(1)

    return cold, warm


# ─── IDEMPOTENCIA ─────────────────────────────────────────────────────────────
def load_completed():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE) as f:
        return set(l.strip() for l in f if l.strip())


def mark_completed(filename):
    with open(LOG_FILE, "a") as f:
        f.write(filename + "\n")


def count_seqs(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 10:
        return 0
    with open(filepath, "rb") as f:
        return f.read().count(b">")


# ─── SESIÓN HTTP ──────────────────────────────────────────────────────────────
def make_session():
    s = requests.Session()
    retry = Retry(total=6, backoff_factor=2.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"Accept": "text/plain"})
    return s


# ─── DESCARGA CON PAGINACIÓN POR CURSOR ───────────────────────────────────────
def fetch_sequences(session, taxa_list, ec_code, label, ec_name):
    filename  = f"{label}_{ec_name}.fasta"
    out_file  = os.path.join(RAW_DIR, filename)
    completed = load_completed()

    if filename in completed:
        n = count_seqs(out_file)
        print(f"  ⏭️  [{label}_{ec_name}] Ya completado ({n:,} seqs) — omitido.")
        return n

    if os.path.exists(out_file):
        print(f"  ♻️  [{label}_{ec_name}] Archivo previo sin completar → re-descargando.")
        os.remove(out_file)

    taxa_query = " OR ".join(f"(taxonomy_id:{t['taxon_id']})" for t in taxa_list)
    full_query = f"(ec:{ec_code}) AND ({taxa_query})"
    url        = "https://rest.uniprot.org/uniprotkb/search"
    params     = {"query": full_query, "format": "fasta", "size": PAGE_SIZE}

    total_written  = 0
    page_num       = 0
    next_cursor    = None
    error_occurred = False

    with open(out_file, "wb") as fasta_out:
        while total_written < MAX_PER_FILE:
            if next_cursor:
                params["cursor"] = next_cursor
            elif "cursor" in params:
                del params["cursor"]

            try:
                resp = session.get(url, params=params, timeout=90)
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f"  ❌ HTTP {resp.status_code} página {page_num}: {e}")
                error_occurred = True
                break
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Red página {page_num}: {e}")
                error_occurred = True
                break

            content = resp.content
            if not content or len(content) < 5:
                break

            fasta_out.write(content)
            total_written += content.count(b">")
            page_num      += 1

            link = resp.headers.get("Link", "")
            if 'rel="next"' in link:
                try:
                    raw_url     = link.split(";")[0].strip().strip("<>")
                    qs          = raw_url.split("?", 1)[1] if "?" in raw_url else ""
                    parts       = [p for p in qs.split("&") if p.startswith("cursor=")]
                    next_cursor = parts[0].split("=", 1)[1] if parts else None
                    if not next_cursor:
                        break
                except (IndexError, ValueError):
                    break
            else:
                break

            time.sleep(SLEEP_PAGE)

    size_kb = os.path.getsize(out_file) / 1024 if os.path.exists(out_file) else 0

    if error_occurred and total_written == 0:
        if os.path.exists(out_file):
            os.remove(out_file)
        print(f"  ❌ [{label}_{ec_name}] Falla total. Reintenta.")
        return 0

    if total_written == 0:
        print(f"  ⚠️  [{label}_{ec_name}] 0 resultados. Verifica el archivo de configuración.")
        if os.path.exists(out_file):
            os.remove(out_file)
        return 0

    mark_completed(filename)
    print(f"  ✅ [{label}_{ec_name}] {total_written:,} seqs  |  {size_kb:,.0f} KB")
    return total_written


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    cold_taxa, warm_taxa = load_taxa()

    print("\n" + "=" * 70)
    print("  PSYCHROSCAN — Descarga Enzimas Industriales")
    print("=" * 70)
    print(f"  Psicrófilos  : {len(cold_taxa)} taxones")
    print(f"  Mesófilos    : {len(warm_taxa)} taxones")
    print(f"  Clases EC    : {len(EC_CLASSES)}")
    print(f"  Máx/archivo  : {MAX_PER_FILE:,} seqs")
    print(f"  Destino      : {RAW_DIR}/")
    print(f"  Idempotente  : Archivos completados se omiten en re-ejecución")
    print("=" * 70 + "\n")

    session = make_session()

    for ec_code, ec_name in EC_CLASSES:
        print(f"── EC {ec_code}  ({ec_name}) " + "─" * 35)
        fetch_sequences(session, cold_taxa, ec_code, "Cold", ec_name)
        fetch_sequences(session, warm_taxa, ec_code, "Warm", ec_name)
        print()
        time.sleep(SLEEP_FILE)

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  RESUMEN FINAL")
    print("=" * 70)
    completed     = load_completed()
    total_size_kb = 0
    total_seqs    = 0
    cold_total    = 0
    warm_total    = 0

    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".fasta"):
            continue
        fpath   = os.path.join(RAW_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        n_seqs  = count_seqs(fpath)
        icon    = "✅" if fname in completed else "⚠️ "
        print(f"  {icon}  {fname:<42}  {n_seqs:>5,} seqs  {size_kb:>7,.0f} KB")
        total_size_kb += size_kb
        total_seqs    += n_seqs
        if fname.startswith("Cold_"):
            cold_total += n_seqs
        elif fname.startswith("Warm_"):
            warm_total += n_seqs

    ratio = warm_total / max(cold_total, 1)
    print(f"\n  TOTAL: {total_seqs:,} seqs  |  {total_size_kb/1024:.2f} MB")
    print(f"  ❄️  Cold: {cold_total:,}  |  🌱 Warm: {warm_total:,}  |  Ratio: {ratio:.1f}x")
    if ratio <= 5:
        print("  ✅ Ratio ≤ 5x — sin undersampling agresivo en 05.")
    else:
        print("  ⚠️  Ratio > 5x — 05 aplicará undersampling automático.")
    print("\n  Siguiente paso → 03_feature_extraction.py\n")


if __name__ == "__main__":
    main()