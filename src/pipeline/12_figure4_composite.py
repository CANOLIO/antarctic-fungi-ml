import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

FIGURES_DIR = os.path.join("results", "figures")
GENOMES_DIR = os.path.join("data", "new_genomes")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLD_COLOR = '#4a90d9'
WARM_COLOR = '#e07b54'
HEADER_BG  = '#2c3e50'
SIG_COLOR  = '#1e8449'


def build_figure():
    fig = plt.figure(figsize=(18, 7))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.06)

    # Panel A: tabla
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis('off')

    col_labels = ['Organism', 'N', 'K', 'Top 15', 'Fold', 'p (Fisher)', 'PPI']
    cell_data  = [
        ['P. chrysogenum (mesophile)',         '12,562', '93', '0', '0.0x', '1.00',    '45.1'],
        ['P. haloplanktis TAC125 (psychro.)',  '3,484',  '43', '1', '5.3x', '0.17',    '63.3+'],
        ['P. haloplanktis OX:228 (psychro.)',  '3,650',  '39', '2', '12.5x','0.011*',  '63.3+'],
    ]

    tbl = ax_table.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.32, 0.10, 0.07, 0.09, 0.09, 0.12, 0.09],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 3.5)

    # Header style
    for col in range(len(col_labels)):
        cell = tbl[0, col]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
        cell.set_edgecolor('white')

    # Row styles
    row_bg    = ['#fdf2f8', '#eaf4fb', '#eaf4fb']
    row_color = [WARM_COLOR, COLD_COLOR, COLD_COLOR]
    for row in range(1, 4):
        for col in range(len(col_labels)):
            cell = tbl[row, col]
            cell.set_facecolor(row_bg[row - 1])
            cell.set_edgecolor('#dee2e6')
            if col == 0:
                cell.set_text_props(color=row_color[row - 1], fontweight='bold')
            if col == 5 and row == 3:
                cell.set_text_props(color=SIG_COLOR, fontweight='bold')

    ax_table.set_title('Retrospective Benchmark — Hydrolytic Enzyme Recovery',
                       fontsize=11, fontweight='bold', color='#2c3e50', pad=14)
    ax_table.text(
        0.01, 0.02,
        '* p < 0.05 (Fisher exact, one-tailed)   + PPI: >50 psychrophile, 30-50 psychrotrophic, <30 mesophile\n'
        'N = proteome size;  K = annotated hydrolytic enzymes;  Top 15 = hydrolytic enzymes in top-15 predictions',
        transform=ax_table.transAxes,
        fontsize=7.5, color='#5d6d7e', va='bottom', linespacing=1.6)

    # Panel B: dendrograma
    ax_dend   = fig.add_subplot(gs[1])
    dend_path = os.path.join(GENOMES_DIR,
                             'Pseudoalteromonas_haloplanktis_TAC125_PPI_context.png')
    if os.path.exists(dend_path):
        img = mpimg.imread(dend_path)
        ax_dend.imshow(img, aspect='auto')
        ax_dend.axis('off')
    else:
        ax_dend.text(0.5, 0.5,
                     'Dendrograma no encontrado.\nCorre: python src/09_predict_new_genome.py --no-pfam',
                     ha='center', va='center', fontsize=10, color='#e74c3c',
                     transform=ax_dend.transAxes)
        ax_dend.axis('off')

    ax_dend.set_title(
        'Proteome-level Positioning — P. haloplanktis TAC125\n'
        'Biochemical distance dendrogram (Ward linkage, 431 features, 35 ref. organisms)',
        fontsize=10, fontweight='bold', pad=8, color='#2c3e50')

    fig.text(0.02, 0.96, 'A', fontsize=16, fontweight='bold', color='#2c3e50')
    fig.text(0.52, 0.96, 'B', fontsize=16, fontweight='bold', color='#2c3e50')

    out = os.path.join(FIGURES_DIR, '12_Figure4_Composite.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Figura 4 guardada: {out}")


if __name__ == "__main__":
    build_figure()