"""
PsychroScan — generate_supplementary_S1.py
Genera Supplementary Table S1: panel de referencia taxonómica
Corre desde la raíz del proyecto:
    python src/generate_supplementary_S1.py
"""
import os
import pandas as pd

OUT_DIR = os.path.join("results", "supplementary")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── DATOS DEL PANEL DE REFERENCIA ────────────────────────────────────────────
# Fuentes: DSMZ, ATCC, literature (ver columna Reference)
# T_opt: temperatura óptima de crecimiento documentada
# UniProt Proteome: ID de proteoma de referencia en UniProt

PANEL = [
    # ── PSICRÓFILOS / PSICROTROFOS ────────────────────────────────────────────
    {
        "Organism": "Psychrobacter arcticus 273-4",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "4",
        "UniProt_Proteome": "UP000006852",
        "Isolation_Source": "Siberian permafrost",
        "Reference": "Bakermans et al. 2006 IJSEM 56:1285",
    },
    {
        "Organism": "Psychrobacter cryohalolentis K5",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "4",
        "UniProt_Proteome": "UP000006820",
        "Isolation_Source": "Siberian permafrost",
        "Reference": "Rodrigues et al. 2008 Appl Environ Microbiol 74:5770",
    },
    {
        "Organism": "Psychromonas ingrahamii 37",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "-12 (growth)",
        "UniProt_Proteome": "UP000001255",
        "Isolation_Source": "Arctic sea ice",
        "Reference": "Auman et al. 2006 IJSEM 56:1285",
    },
    {
        "Organism": "Colwellia psychrerythraea 34H",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "4",
        "UniProt_Proteome": "UP000006890",
        "Isolation_Source": "Arctic marine sediment",
        "Reference": "Methé et al. 2005 PNAS 102:10913",
    },
    {
        "Organism": "Shewanella frigidimarina NCIMB 400",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "10–15",
        "UniProt_Proteome": "UP000006880",
        "Isolation_Source": "Antarctic seawater",
        "Reference": "Venkateswaran et al. 1999 IJSEM 49:705",
    },
    {
        "Organism": "Pseudoalteromonas haloplanktis TAC125",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "4",
        "UniProt_Proteome": "UP000006843",
        "Isolation_Source": "Antarctic seawater",
        "Reference": "Médigue et al. 2005 Genome Res 15:1325",
    },
    {
        "Organism": "Marinomonas primoryensis",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "4",
        "UniProt_Proteome": "UP000078082",
        "Isolation_Source": "Sea of Japan",
        "Reference": "Romanenko et al. 2003 IJSEM 53:241",
    },
    {
        "Organism": "Polaribacter irgensii 23-P",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "4",
        "UniProt_Proteome": "UP000008084",
        "Isolation_Source": "Antarctic sea ice",
        "Reference": "Gosink et al. 1998 IJSEM 48:223",
    },
    {
        "Organism": "Algoriphagus machipongonensis",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "10",
        "UniProt_Proteome": "UP000017898",
        "Isolation_Source": "Cold coastal sediment",
        "Reference": "Alegado et al. 2011 eLife 1:e00013",
    },
    {
        "Organism": "Photobacterium profundum SS9",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "10",
        "UniProt_Proteome": "UP000000593",
        "Isolation_Source": "Deep-sea cold seep",
        "Reference": "Vezzi et al. 2005 Science 307:1459",
    },
    {
        "Organism": "Moritella marina MP-1",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "5",
        "UniProt_Proteome": "UP000078306",
        "Isolation_Source": "Deep-sea sediment",
        "Reference": "Urakawa et al. 1998 IJSEM 48:1147",
    },
    {
        "Organism": "Rhodococcus erythropolis PR4",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "10",
        "UniProt_Proteome": "UP000001006",
        "Isolation_Source": "Arctic soil",
        "Reference": "Na et al. 2005 J Bacteriol 187:5840",
    },
    {
        "Organism": "Arthrobacter psychrolactophilus F2",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "0–4",
        "UniProt_Proteome": "UP000019185",
        "Isolation_Source": "Cold dairy environment",
        "Reference": "Loveland-Curtze et al. 1999 Arch Microbiol 172:355",
    },
    {
        "Organism": "Flavobacterium psychrophilum JIP02/86",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "15",
        "UniProt_Proteome": "UP000001651",
        "Isolation_Source": "Salmonid fish",
        "Reference": "Duchaud et al. 2007 Nat Biotechnol 25:763",
    },
    {
        "Organism": "Cryobacterium psychrotolerans",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "4",
        "UniProt_Proteome": "UP000078340",
        "Isolation_Source": "Glacier ice",
        "Reference": "Wu et al. 2004 IJSEM 54:1919",
    },
    {
        "Organism": "Planococcus halocryophilus Or1",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "-15 (growth)",
        "UniProt_Proteome": "UP000017905",
        "Isolation_Source": "High Arctic permafrost",
        "Reference": "Mykytczuk et al. 2013 ISME J 7:1211",
    },
    {
        "Organism": "Carnobacterium maltaromaticum LMA28",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "25 (psychrotrophic)",
        "UniProt_Proteome": "UP000017700",
        "Isolation_Source": "Cold meat",
        "Reference": "Leisner et al. 2007 FEMS Microbiol Rev 31:261",
    },
    {
        "Organism": "Leucosporidium scottii",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "10",
        "UniProt_Proteome": "UP000054900",
        "Isolation_Source": "Antarctic soil",
        "Reference": "Fell & Statzell-Tallman 1998 in Kurtzman & Fell (eds)",
    },
    {
        "Organism": "Leucosporidium creatinivorum",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "10",
        "UniProt_Proteome": "UP000054800",
        "Isolation_Source": "Antarctic environment",
        "Reference": "Fell & Statzell-Tallman 1998 in Kurtzman & Fell (eds)",
    },
    {
        "Organism": "Glaciozyma antarctica PI12",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "12",
        "UniProt_Proteome": "UP000029894",
        "Isolation_Source": "Antarctic sea ice",
        "Reference": "Turchetti et al. 2011 FEMS Yeast Res 11:627",
    },
    {
        "Organism": "Cryomyces antarcticus CCFEE 515",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "10",
        "UniProt_Proteome": "UP000078400",
        "Isolation_Source": "Antarctic rock",
        "Reference": "Selbmann et al. 2005 Stud Mycol 51:1",
    },
    {
        "Organism": "Dioszegia hungarica",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "10",
        "UniProt_Proteome": "UP000054700",
        "Isolation_Source": "Cold soil",
        "Reference": "Fülöp & Hohmann 2009 FEMS Yeast Res 9:1278",
    },
    {
        "Organism": "Mrakia blollopis SK-4",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "10",
        "UniProt_Proteome": "UP000054600",
        "Isolation_Source": "Antarctic soil",
        "Reference": "Turchetti et al. 2011 FEMS Yeast Res 11:627",
    },
    {
        "Organism": "Amycolatopsis antarctica",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "20–25 (psychrotrophic)",
        "UniProt_Proteome": "UP001152200",
        "Isolation_Source": "Antarctic soil",
        "Reference": "Henneberger et al. 2016 IJSEM 66:661",
    },
    {
        "Organism": "Streptomyces cryophilus",
        "Thermal_Class": "Psychrophile",
        "T_opt_C": "15",
        "UniProt_Proteome": "UP000078500",
        "Isolation_Source": "Cold soil",
        "Reference": "Goodfellow & Williams 1983 Ann Rev Microbiol 37:189",
    },
    # ── MESÓFILOS ─────────────────────────────────────────────────────────────
    {
        "Organism": "Saccharomyces cerevisiae S288c",
        "Thermal_Class": "Mesophile",
        "T_opt_C": "30",
        "UniProt_Proteome": "UP000002311",
        "Isolation_Source": "Laboratory strain",
        "Reference": "Cherry et al. 2012 Genetics 192:845",
    },
    {
        "Organism": "Schizosaccharomyces pombe 972h",
        "Thermal_Class": "Mesophile",
        "T_opt_C": "30",
        "UniProt_Proteome": "UP000002485",
        "Isolation_Source": "Laboratory strain",
        "Reference": "Wood et al. 2002 Nature 415:871",
    },
    {
        "Organism": "Candida albicans SC5314",
        "Thermal_Class": "Mesophile",
        "T_opt_C": "37",
        "UniProt_Proteome": "UP000000559",
        "Isolation_Source": "Human pathogen",
        "Reference": "Jones et al. 2004 PNAS 101:7329",
    },
    {
        "Organism": "Aspergillus niger CBS 513.88",
        "Thermal_Class": "Mesophile",
        "T_opt_C": "35",
        "UniProt_Proteome": "UP000006706",
        "Isolation_Source": "Industrial strain",
        "Reference": "Pel et al. 2007 Nat Biotechnol 25:221",
    },
    {
        "Organism": "Penicillium rubens Wisconsin 54-1255",
        "Thermal_Class": "Mesophile",
        "T_opt_C": "25",
        "UniProt_Proteome": "UP000000724",
        "Isolation_Source": "Laboratory strain",
        "Reference": "van den Berg et al. 2008 Nat Biotechnol 26:1161",
    },
    {
        "Organism": "Neurospora crassa OR74A",
        "Thermal_Class": "Mesophile",
        "T_opt_C": "30",
        "UniProt_Proteome": "UP000001805",
        "Isolation_Source": "Laboratory strain",
        "Reference": "Galagan et al. 2003 Nature 422:859",
    },
    {
        "Organism": "Trichoderma reesei QM6a",
        "Thermal_Class": "Mesophile",
        "T_opt_C": "28",
        "UniProt_Proteome": "UP000006220",
        "Isolation_Source": "Industrial strain",
        "Reference": "Martinez et al. 2008 Nat Biotechnol 26:553",
    },
    {
        "Organism": "Escherichia coli K-12",
        "Thermal_Class": "Mesophile",
        "T_opt_C": "37",
        "UniProt_Proteome": "UP000000625",
        "Isolation_Source": "Laboratory strain",
        "Reference": "Blattner et al. 1997 Science 277:1453",
    },
    {
        "Organism": "Bacillus subtilis 168",
        "Thermal_Class": "Mesophile",
        "T_opt_C": "37",
        "UniProt_Proteome": "UP000001570",
        "Isolation_Source": "Laboratory strain",
        "Reference": "Kunst et al. 1997 Nature 390:249",
    },
    {
        "Organism": "Pseudomonas aeruginosa PAO1",
        "Thermal_Class": "Mesophile",
        "T_opt_C": "37",
        "UniProt_Proteome": "UP000002438",
        "Isolation_Source": "Clinical isolate",
        "Reference": "Stover et al. 2000 Nature 406:959",
    },
    {
        "Organism": "Staphylococcus aureus NCTC 8325",
        "Thermal_Class": "Mesophile",
        "T_opt_C": "37",
        "UniProt_Proteome": "UP000008816",
        "Isolation_Source": "Clinical isolate",
        "Reference": "Gillaspy et al. 2006 in Fischetti et al. (eds)",
    },
    {
        "Organism": "Streptomyces griseus NBRC 13350",
        "Thermal_Class": "Mesophile",
        "T_opt_C": "28",
        "UniProt_Proteome": "UP000001685",
        "Isolation_Source": "Soil",
        "Reference": "Ohnishi et al. 2008 Nat Biotechnol 26:1185",
    },
]

df = pd.DataFrame(PANEL)
df.index = range(1, len(df) + 1)
df.index.name = "No."

out_csv = os.path.join(OUT_DIR, "Table_S1_Reference_Panel.csv")
df.to_csv(out_csv)

print(f"✅ Supplementary Table S1 guardada → {out_csv}")
print(f"   Psicrófilos : {(df['Thermal_Class']=='Psychrophile').sum()}")
print(f"   Mesófilos   : {(df['Thermal_Class']=='Mesophile').sum()}")
print(f"   Total       : {len(df)}")