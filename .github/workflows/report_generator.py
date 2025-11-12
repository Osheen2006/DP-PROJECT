
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PyQt5.QtWidgets import QInputDialog, QMessageBox
import random


REPORTS_FOLDER = "reports"

# Ensure the reports folder exists
if not os.path.exists(REPORTS_FOLDER):
    os.makedirs(REPORTS_FOLDER)

def generate_report(parent, image_path, metadata_text):
    """
    Generates a PDF report for the selected image.
    :param parent: reference to the main window (for dialogs)
    :param image_path: selected image file path
    :param metadata_text: multi-line string with metadata
    """
    if not image_path:
        QMessageBox.warning(parent, "No Image Selected", "Please select an image before generating a report.")
        return

    # Ask for author name
    author, ok = QInputDialog.getText(parent, "Author Name", "Enter your name:")
    if not ok or not author.strip():
        return

    img_name = os.path.basename(image_path)
    pdf_name = f"{os.path.splitext(img_name)[0]}_report.pdf"
    pdf_path = os.path.join(REPORTS_FOLDER, pdf_name)

    # Create PDF
    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, 800, "Welding Defect Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Image Name: {img_name}")
    c.drawString(50, 750, f"Author: {author}")

    # Write metadata
    y = 720
    for line in metadata_text.split("\n"):
        c.drawString(50, y, line)
        y -= 18

    c.save()

    QMessageBox.information(parent, "Report Created", f"Report saved at:\n{pdf_path}")


def open_reports_folder():
    """
    Opens the reports folder in file explorer.
    """
    import subprocess
    folder = os.path.abspath(REPORTS_FOLDER)

    if os.name == "nt":  # Windows
        subprocess.Popen(f'explorer "{folder}"')
    else:  # Mac/Linux
        subprocess.Popen(["open", folder])
