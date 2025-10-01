import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QWidget, QGridLayout, QScrollArea
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal


# Custom QLabel that emits a signal when clicked
class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)   # will emit the image path

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.path)


class DRDOGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basic Frontend")
        self.setGeometry(200, 200, 1200, 800)

        self.current_image_path = None
        self.current_pixmap = None
        self.original_size = None
        self.thumbnail_labels = []

        # ================= Main Layout =================
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)

        # ================= Column 0 (Upload + Thumbnails) =================
        self.upload_btn = QPushButton("Upload Folder")
        self.upload_btn.clicked.connect(self.open_folder)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)

        vbox_left = QVBoxLayout()
        vbox_left.addWidget(self.upload_btn)
        vbox_left.addWidget(self.scroll_area)

        section_left = QWidget()
        section_left.setLayout(vbox_left)
        self.layout.addWidget(section_left, 0, 0, 2, 1)

        # ================= Quadrant Right =================
        # Top-Left: Selected Image
        self.image_label = QLabel("Click a thumbnail to show here")
        self.image_label.setStyleSheet("border: 1px solid gray; min-height: 200px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.layout.addWidget(self.image_label, 0, 1)

        # Top-Right: Process
        self.process_btn = QPushButton("Process (Not Implemented)")
        self.output_label = QLabel("Processed output will appear here")
        self.output_label.setStyleSheet("border: 1px solid gray; min-height: 100px;")

        vbox_proc = QVBoxLayout()
        vbox_proc.addWidget(self.process_btn)
        vbox_proc.addWidget(self.output_label)

        section_proc = QWidget()
        section_proc.setLayout(vbox_proc)
        self.layout.addWidget(section_proc, 0, 2)

        # Bottom-Left: Categorize
        self.categorize_btn = QPushButton("Categorize (Repair / Accept)")
        self.categorize_btn.clicked.connect(self.categorize_image)
        self.category_label = QLabel("Category result will appear here")
        self.category_label.setStyleSheet("border: 1px solid gray; min-height: 50px;")

        vbox_cat = QVBoxLayout()
        vbox_cat.addWidget(self.categorize_btn)
        vbox_cat.addWidget(self.category_label)

        section_cat = QWidget()
        section_cat.setLayout(vbox_cat)
        self.layout.addWidget(section_cat, 1, 1)

        # Bottom-Right: Metadata
        self.metadata_label = QLabel("Metadata will appear here")
        self.metadata_label.setStyleSheet("border: 1px solid gray; min-height: 100px;")

        vbox_meta = QVBoxLayout()
        vbox_meta.addWidget(self.metadata_label)

        section_meta = QWidget()
        section_meta.setLayout(vbox_meta)
        self.layout.addWidget(section_meta, 1, 2)

        # Layout proportions
        self.layout.setColumnStretch(0, 1)  # thin left
        self.layout.setColumnStretch(1, 2)
        self.layout.setColumnStretch(2, 2)
        self.layout.setRowStretch(0, 2)
        self.layout.setRowStretch(1, 1)

    # ================= Upload Folder =================
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            # Clear old thumbnails
            for i in reversed(range(self.scroll_layout.count())):
                widget = self.scroll_layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()
            self.thumbnail_labels.clear()

            # Add new thumbnails
            for fname in os.listdir(folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(folder, fname)
                    pixmap = QPixmap(path).scaledToWidth(100, Qt.SmoothTransformation)

                    thumb_label = ClickableLabel(path)
                    thumb_label.setPixmap(pixmap)
                    thumb_label.setStyleSheet("border: 1px solid gray; margin: 2px;")
                    thumb_label.setCursor(Qt.PointingHandCursor)

                    # connect signal to slot
                    thumb_label.clicked.connect(self.on_thumbnail_clicked)

                    self.scroll_layout.addWidget(thumb_label)
                    self.thumbnail_labels.append(thumb_label)

    # ================= Thumbnail Click =================
    def on_thumbnail_clicked(self, path):
        # Reset all borders
        for thumb in self.thumbnail_labels:
            thumb.setStyleSheet("border: 1px solid gray; margin: 2px;")
        # Highlight selected one
        sender = self.sender()
        if isinstance(sender, QLabel):
            sender.setStyleSheet("border: 2px solid blue; margin: 2px;")
        # Load image
        self.load_image(path)

    # ================= Load Image =================
    def load_image(self, path):
        self.current_image_path = path
        self.current_pixmap = QPixmap(path)

        from PIL import Image
        img = Image.open(path)
        self.original_size = img.size

        self.resize_image_to_label()
        self.show_metadata(path)

    def resize_image_to_label(self):
        if self.current_pixmap:
            scaled_pixmap = self.current_pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.update_metadata_display()

    def resizeEvent(self, event):
        self.resize_image_to_label()
        super().resizeEvent(event)

    # ================= Metadata =================
    def show_metadata(self, fname):
        from PIL import Image
        img = Image.open(fname)

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
        if self.current_pixmap and self.original_size:
            displayed_w = self.image_label.width()
            displayed_h = self.image_label.height()
            displayed_size = f"Displayed Size: {displayed_w} x {displayed_h} px"
            self.metadata_label.setText(self.base_metadata_text + "\n" + displayed_size)

    # ================= Categorization =================
    def categorize_image(self):
        if not self.current_image_path:
            self.category_label.setText("No image selected!")
            return

        import random
        category = random.choice(["Repair", "Accept"])
        self.category_label.setText(f"Result: {category}")


def main():
    app = QApplication(sys.argv)
    window = DRDOGUI()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
