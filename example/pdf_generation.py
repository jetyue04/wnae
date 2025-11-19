import re
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import KeepTogether

def add_config_section(cfg, elements, styles, mono_style, indent=0):
    """Recursively add config dictionary to PDF elements."""
    for k, v in cfg.items():
        if isinstance(v, dict):
            elements.append(Paragraph(" " * indent + f"<b>{k}:</b>", mono_style))
            add_config_section(v, elements, styles, mono_style, indent + 4)
        else:
            elements.append(Paragraph(" " * indent + f"{k}: {v}", mono_style))

def get_plot_path(feature_dir, epoch):
    """Get the file path for the plot image corresponding to the given epoch."""
    path = feature_dir / f'epoch_{epoch}.png'
    return path if path.exists() else None

def create_report(output_dir, config):
    # Ensure Paths are correct
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if not output_dir.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    plots_dir = output_dir / 'sample_feature_1D_hist'
    training_loss_path = output_dir / 'train_history.png'
    info_path = output_dir / "info.txt"

    # output pdf path
    pdf_path = output_dir / "report.pdf"

    ## Extract best epoch:
    best_epoch = None
    if info_path.exists():
        with open(info_path, "r") as f:
            text = f.read()
            match = re.search(r"Best\s*epoch:\s*(\d+)", text)
            if match:
                best_epoch = int(match.group(1))
                print(f"Found best epoch: {best_epoch}")
    else:
        print("info.txt not found — skipping best epoch detection.")


    
    ## Find features and epochs
    feature_dirs = sorted( [d for d in plots_dir.iterdir() if d.is_dir() and "feature" in d.name],
    key=lambda d: int(re.search(r'\d+', d.name).group())
    )
    
    first_feature_dir = feature_dirs[0]
    all_epochs = sorted([
        int(f.stem.replace("epoch_", ""))
        for f in first_feature_dir.glob("epoch_*.png")
    ])

    interval = config['plot']['interval']
    selected_epochs = {e for e in all_epochs if e % interval == 0} | {max(all_epochs)}
    if best_epoch is not None:
        # Ensure best_epoch is included and appears first
        epochs = [best_epoch] + sorted(selected_epochs - {best_epoch})
    else:
        epochs = sorted(selected_epochs)

    print(f"Final epoch order (best epoch first): {epochs}")

    # ---- Build the document ----
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenteredTitle", parent=styles["Title"], alignment=1))
    styles.add(ParagraphStyle(name="Subheading", fontSize=14, spaceAfter=12, leading=18))
    styles.add(ParagraphStyle(name="Mono", parent=styles["Normal"], fontName="Courier", fontSize=9, leading=12,))
    mono_style = styles["Mono"]
    elements = []

    # ---- Title Page ----
    elements.append(Paragraph(f"Results for WNAE Run {output_dir}", styles["CenteredTitle"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"<b>Best Epoch:</b> {best_epoch}", styles["Subheading"]))
    elements.append(Spacer(1, 0.2 * inch))

    # ---- Include training loss plot if it exists ----
    elements.append(Paragraph("<b>Loss and AUC plots</b>", styles["Subheading"]))
    if training_loss_path.exists():
        img = Image(str(training_loss_path), width=6.5 * inch, height=4 * inch)
        elements.append(img)
    else:
        elements.append(Paragraph("Training loss plot not found.", styles["Normal"]))

    elements.append(Paragraph("<b>Configuration Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    if config:
        add_config_section(config, elements, styles, mono_style)
    else:
        elements.append(Paragraph("No configuration found.", styles["Normal"]))
    elements.append(PageBreak())

    # ---- Feature Plots ----
    num_cols = 3  # number of columns
    col_width = 2.5*inch

    for epoch in epochs:
        # Epoch heading
        epoch_title = f"Epoch {epoch}"
        if epoch == best_epoch:
            epoch_title = "Best Epoch: " + epoch_title
        elements.append(Paragraph(f"<b>{epoch_title}</b>", styles["Subheading"]))
        elements.append(Spacer(1, 6))

        # Collect images for this epoch
        table_data = []
        row_cells = []

        for i, feature_dir in enumerate(feature_dirs):
            feature_name = feature_dir.name
            img_path = get_plot_path(feature_dir, epoch)
            if img_path:
                img = Image(str(img_path), width=2.5*inch, height=1.8*inch)
                cell = [img, Paragraph(feature_name, styles["Normal"])]
            else:
                cell = [Paragraph(f"Missing\n{feature_name}", styles["Normal"])]

            row_cells.append(cell)

            # If row is full, append to table_data and start new row
            if (i + 1) % num_cols == 0:
                table_data.append(row_cells)
                row_cells = []

        # Append any remaining cells
        if row_cells:
            # Fill empty cells to match number of columns
            while len(row_cells) < num_cols:
                row_cells.append("")
            table_data.append(row_cells)

        # Create table
        table = Table(table_data, colWidths=[col_width]*num_cols)
        table.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
    
        elements.append(table)
    # ---- Build PDF ----
    doc.build(elements)
    print(f"PDF report saved to: {pdf_path}")




    