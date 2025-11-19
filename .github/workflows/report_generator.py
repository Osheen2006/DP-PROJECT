import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel,
    QWidget, QVBoxLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QLinearGradient, QColor, QBrush, QMovie
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRect, QEasingCurve

from main import GUI  # ensure main.py with GUI class exists


class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welding Defect Detection System - Loading")
        self.setGeometry(50, 100, 1700, 1000)
        self.setMinimumSize(1200, 700)

        # Gradient background
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 80, 900)
        gradient.setColorAt(0.0, QColor("#AFCEE8"))
        gradient.setColorAt(1.0, QColor("#E8EEF2"))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)

        # Central widget
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)

        # Card Frame
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.25);
                border-radius: 60px;
                padding: 40px;
                border: 1px solid rgba(0, 0, 0, 0.1);
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignCenter)

        # Logo
        self.logo_label = QLabel()
        pixmap = QPixmap("C:\\Users\\Lenovo\\Desktop\\DP\\DP project\\DP project\\logooo.png")
        self.logo_label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logo_label)

        # Title
        title = QLabel("GUI for Welding Defect Detection ")
        title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #0C2340; margin-top: 10px;")
        layout.addWidget(title)

                
        # Loading GIF
        self.loading_label = QLabel()
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_movie = QMovie("C:\\Users\\Lenovo\\Desktop\\DP\\DP project\\DP project\\loaad2.gif")  # <-- your GIF file here
        self.loading_movie.setScaledSize(QPixmap("C:\\Users\\Lenovo\\Desktop\\DP\\DP project\\DP project\\loaad2.gif").size())
        self.loading_label.setMovie(self.loading_movie)
        self.loading_movie.start()
        layout.addWidget(self.loading_label)
        
        self.setCentralWidget(central_widget)

        

   

        # Timer for transition
        QTimer.singleShot(2000, self.open_main_page)  # auto open after 4 seconds

    def open_main_page(self):
        self.main_window = GUI()
        
        self.main_window.show()
        self.main_window.showMaximized()
        
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StartWindow()
    window.showMaximized()
    window.show()
    sys.exit(app.exec())
