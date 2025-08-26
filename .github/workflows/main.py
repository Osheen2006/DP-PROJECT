import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap


class DRDOGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basic Frontend")
        self.setGeometry(200, 200, 1000, 800)

        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Upload button
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.open_file)
        self.layout.addWidget(self.upload_btn)

        # Label to show uploaded image
        self.image_label = QLabel("Uploaded image will appear here")
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.layout.addWidget(self.image_label)

        # Process button (placeholder)
        self.process_btn = QPushButton("Process (Not Implemented)")
        self.layout.addWidget(self.process_btn)

        # Output label
        self.output_label = QLabel("Processed output will appear here")
        self.output_label.setStyleSheet("border: 1px solid gray;")
        self.layout.addWidget(self.output_label)

    def open_file(self):
        """Open file dialog to select an image and show preview."""
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            pixmap = QPixmap(fname).scaled(400, 400)  # resize to fit
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)


def main():
    app = QApplication(sys.argv)
    window = DRDOGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


