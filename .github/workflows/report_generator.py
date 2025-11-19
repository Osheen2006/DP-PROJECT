import os
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from PyQt5.QtWidgets import QInputDialog, QMessageBox
import subprocess

REPORTS_FOLDER = "reports"
if not os.path.exists(REPORTS_FOLDER):
    os.makedirs(REPORTS_FOLDER)

def _draw_text_block(c, title, content, x, y):
    """Helper to draw a simple text block with naive wrapping/page breaks."""
    if title:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, title)
        y -= 16

    c.setFont("Helvetica", 10)
    lines = content.split("\n")
    for line in lines:
        if y < 120:
            c.showPage()
            y = A4[1] - 60
            c.setFont("Helvetica", 10)
        c.drawString(x, y, line)
        y -= 14
    return y - 8

def _draw_image_and_caption(c, path, x, y, max_h=2.5 * inch, caption=""):
    """Draw image at (x,y) with caption underneath. Returns new Y after drawing."""
    if not path or not os.path.exists(path):
        c.setFont("Helvetica", 9)
        c.drawString(x, y - 12, f"[Image not available: {os.path.basename(path) if path else 'N/A'}]")
        return y - (max_h + 20)

    try:
        img_h = max_h
        img_w = max_h * 1.33
        c.drawImage(path, x, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, anchor='n')
        if caption:
            c.setFont("Helvetica", 9)
            c.drawString(x, y - img_h - 12, caption)
            return y - img_h - 24
        return y - img_h - 10
    except Exception:
        c.setFont("Helvetica", 9)
        c.drawString(x, y - 12, f"[Failed to render image: {os.path.basename(path)}]")
        return y - (max_h + 20)

def _coords_to_strings(coords):
    """Return tuple (coords_str, size_str) given coords (list/tuple)."""
    try:
        if isinstance(coords, (list, tuple)) and len(coords) == 4:
            x1, y1, x2, y2 = map(float, coords)
            w = max(0, int(round(x2 - x1)))
            h = max(0, int(round(y2 - y1)))
            coords_str = f"{int(round(x1))}, {int(round(y1))}, {int(round(x2))}, {int(round(y2))}"
            size_str = f"{w} x {h}"
            return coords_str, size_str
    except Exception:
        pass
    return "N/A", "N/A"

def generate_report(parent, image_path, metadata_text, detection_image_path=None, segmentation_image_path=None, bounding_boxes=None):
    """Generates a professional PDF report including a defect table with size column."""
    if not image_path:
        QMessageBox.warning(parent, "No Image Selected", "Please select an image before generating a report.")
        return

    author, ok = QInputDialog.getText(parent, "Author Name", "Enter your name for the report:")
    if not ok or not author.strip():
        return

    img_base_name = os.path.basename(image_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = f"{os.path.splitext(img_base_name)[0]}_REPORT_{timestamp}.pdf"
    pdf_path = os.path.join(REPORTS_FOLDER, pdf_name)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    page_w, page_h = A4
    margin = 50
    y = page_h - margin

    # Header
    c.setFont("Helvetica-Bold", 22)
    c.setFillColorRGB(0.08, 0.08, 0.35)
    c.drawString(margin, y, "Automated Defect Analysis Report")
    y -= 22
    c.setStrokeColorRGB(0.08, 0.08, 0.35)
    c.line(margin, y, page_w - margin, y)
    y -= 26

    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(margin, y, f"Image File: {img_base_name}")
    c.drawString(page_w - 240, y, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 14
    c.drawString(margin, y, f"Report Author: {author}")
    y -= 20

    # 1. Metadata
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "1. File & System Information")
    y -= 18
    y = _draw_text_block(c, "", metadata_text, margin, y)
    y -= 12

    # 2. Visual Results
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "2. Visual Inspection and Analysis")
    y -= 18
    visual_top_y = y
    MAX_IMG_H = 2.4 * inch
    y = _draw_image_and_caption(c, image_path, margin, y, max_h=MAX_IMG_H, caption="Figure 2.1: Original Image")
    _ = _draw_image_and_caption(c, detection_image_path, margin + 4.6 * inch, visual_top_y, max_h=MAX_IMG_H, caption="Figure 2.2: Detection (Bounding Boxes)")
    if segmentation_image_path and os.path.exists(segmentation_image_path):
        y = _draw_image_and_caption(c, segmentation_image_path, margin, y, max_h=MAX_IMG_H, caption="Figure 2.3: Segmentation (Coloured Mask)")
    else:
        y -= 10
    y -= 12

    # 3. Detected Defects Table (with Size)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "3. Detected Defects Table")
    y -= 18

    # Table layout coordinates
    table_x = margin
    col_class_x = table_x
    col_conf_x = table_x + 140
    col_coords_x = table_x + 240
    col_size_x = table_x + 470
    row_h = 16

    # Header row
    c.setFont("Helvetica-Bold", 10)
    c.drawString(col_class_x, y, "Defect Type")
    c.drawString(col_conf_x, y, "Confidence")
    c.drawString(col_coords_x, y, "Coordinates (X1, Y1, X2, Y2)")
    c.drawString(col_size_x, y, "Size (w × h)")
    y -= row_h
    c.line(table_x, y + 6, page_w - margin, y + 6)
    y -= 6

    c.setFont("Helvetica", 10)
    used_any = False

    if bounding_boxes and isinstance(bounding_boxes, (list, tuple)):
        # iterate and print rows
        for bb in bounding_boxes:
            # Skip placeholder/simulated outputs if marked as such
            if isinstance(bb, dict) and bb.get("class") == "simulated_defect":
                continue

            if y < 120:
                c.showPage()
                y = page_h - margin
                c.setFont("Helvetica-Bold", 10)
                c.drawString(col_class_x, y, "Defect Type")
                c.drawString(col_conf_x, y, "Confidence")
                c.drawString(col_coords_x, y, "Coordinates (X1, Y1, X2, Y2)")
                c.drawString(col_size_x, y, "Size (w × h)")
                y -= row_h
                c.line(table_x, y + 6, page_w - margin, y + 6)
                y -= 6
                c.setFont("Helvetica", 10)

            # Defensive extraction of fields
            cls = bb.get("class", "N/A") if isinstance(bb, dict) else "N/A"
            conf = bb.get("confidence", "N/A") if isinstance(bb, dict) else "N/A"
            coords = bb.get("coords", []) if isinstance(bb, dict) else []

            coords_str, size_str = _coords_to_strings(coords)

            c.drawString(col_class_x, y, str(cls))
            c.drawString(col_conf_x, y, str(conf))
            c.drawString(col_coords_x, y, coords_str)
            c.drawString(col_size_x, y, size_str)
            y -= row_h
            used_any = True

    if not used_any:
        c.setFont("Helvetica", 10)
        c.drawString(table_x, y, "No defects detected or model output flagged as simulated/placeholder.")
        y -= row_h

    # Save PDF
    c.showPage()
    c.save()

    QMessageBox.information(parent, "Report Created", f"Professional Report saved at:\n{os.path.abspath(pdf_path)}")

def open_reports_folder():
    folder = os.path.abspath(REPORTS_FOLDER)
    try:
        if os.name == "nt":
            subprocess.Popen(f'explorer "{folder}"')
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    except Exception:
        try:
            import webbrowser
            webbrowser.open(folder)
        except Exception:
            pass
