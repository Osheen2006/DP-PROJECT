
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QWidget, QGridLayout
)
from PyQt5.QtGui import QPixmap


class DRDOGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basic Frontend")
        self.setGeometry(200, 200, 1000, 800)

        # Keep track of uploaded image path and pixmap
        self.current_image_path = None
        self.current_pixmap = None
        self.original_size = None  # store original image resolution

        # Main layout: Grid
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)

        # ================= Section 1 (Top-Left: Image) =================
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.open_file)

        self.image_label = QLabel("Uploaded image will appear here")
        self.image_label.setStyleSheet("border: 1px solid gray; min-height: 200px;")
        self.image_label.setScaledContents(True)

        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.upload_btn)
        vbox1.addWidget(self.image_label)

        section1 = QWidget()
        section1.setLayout(vbox1)
        self.layout.addWidget(section1, 0, 0)

        # ================= Section 2 (Top-Right: Process) =================
        self.process_btn = QPushButton("Process (Not Implemented)")
        self.output_label = QLabel("Processed output will appear here")
        self.output_label.setStyleSheet("border: 1px solid gray; min-height: 100px;")

        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.process_btn)
        vbox2.addWidget(self.output_label)

        section2 = QWidget()
        section2.setLayout(vbox2)
        self.layout.addWidget(section2, 0, 1)

        # ================= Section 3 (Bottom-Left: Categorize) =================
        self.categorize_btn = QPushButton("Categorize (Repair / Accept)")
        self.categorize_btn.clicked.connect(self.categorize_image)
        self.category_label = QLabel("Category result will appear here")
        self.category_label.setStyleSheet("border: 1px solid gray; min-height: 50px;")

        vbox3 = QVBoxLayout()
        vbox3.addWidget(self.categorize_btn)
        vbox3.addWidget(self.category_label)

        section3 = QWidget()
        section3.setLayout(vbox3)
        self.layout.addWidget(section3, 1, 0)

        # ================= Section 4 (Bottom-Right: Metadata) =================
        self.metadata_label = QLabel("Metadata will appear here")
        self.metadata_label.setStyleSheet("border: 1px solid gray; min-height: 100px;")

        vbox4 = QVBoxLayout()
        vbox4.addWidget(self.metadata_label)

        section4 = QWidget()
        section4.setLayout(vbox4)
        self.layout.addWidget(section4, 1, 1)

        # ====== Set stretch factors (Section 1 grows more) ======
        self.layout.setColumnStretch(0, 2)  # left side (image/categorize) bigger
        self.layout.setColumnStretch(1, 1)  # right side adjusts
        self.layout.setRowStretch(0, 2)     # top row (image/process) bigger
        self.layout.setRowStretch(1, 1)     # bottom row adjusts

    def open_file(self):
        """Open file dialog to select an image and show preview + metadata."""
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if fname:
            self.current_image_path = fname
            self.current_pixmap = QPixmap(fname)

            # Load original resolution
            from PIL import Image
            img = Image.open(fname)
            self.original_size = img.size  # (width, height)

            # Show image and metadata
            self.resize_image_to_label()
            self.show_metadata(fname)

    def resize_image_to_label(self):
        """Resize current pixmap to fit the image_label while keeping aspect ratio."""
        if self.current_pixmap:
            scaled_pixmap = self.current_pixmap.scaled(
                self.image_label.size(),
                aspectRatioMode=1  # Keep aspect ratio
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.update_metadata_display()

    def resizeEvent(self, event):
        """Handle window resizing and adjust image & metadata accordingly."""
        self.resize_image_to_label()
        super().resizeEvent(event)

    def show_metadata(self, fname):
        """Store base metadata of the uploaded image."""
        from PIL import Image
        img = Image.open(fname)

        # Get file size
        file_size_bytes = os.path.getsize(fname)
        if file_size_bytes < 1024:
            file_size_str = f"{file_size_bytes} B"
        elif file_size_bytes < 1024 * 1024:
            file_size_str = f"{file_size_bytes/1024:.2f} KB"
        else:
            file_size_str = f"{file_size_bytes/(1024*1024):.2f} MB"

        self.base_metadata_text = (
            f"File: {fname}\n"
            f"Format: {img.format}\n"
            f"Original Size: {img.size[0]} x {img.size[1]} px\n"
            f"Mode: {img.mode}\n"
            f"File Size: {file_size_str}"
        )
        self.update_metadata_display()

    def update_metadata_display(self):
        """Update metadata label to also include displayed size."""
        if self.current_pixmap and self.original_size:
            displayed_w = self.image_label.width()
            displayed_h = self.image_label.height()
            displayed_size = f"Displayed Size: {displayed_w} x {displayed_h} px"
            self.metadata_label.setText(self.base_metadata_text + "\n" + displayed_size)

    def categorize_image(self):
        """Dummy categorization: decide 'Repair' or 'Accept'."""
        if not self.current_image_path:
            self.category_label.setText("No image uploaded!")
            return

        import random
        category = random.choice(["Repair", "Accept"])
        self.category_label.setText(f"Result: {category}")


def main():
    app = QApplication(sys.argv)
    window = DRDOGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
