import sys
import os
import json 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QWidget, QGridLayout, QScrollArea,
    QAction, QMenu, QMessageBox 
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

# Import PIL/Pillow for image processing and metadata
# This must be installed: pip install Pillow
from PIL import Image 

# Custom QLabel that emits a signal when clicked
class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)     # will emit the image path

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.path)


class DRDOGUI(QMainWindow):
    # File to store the application state
    SAVE_FILE = "app_state.json"
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basic Frontend")
        self.setGeometry(200, 200, 1200, 800)

        self.current_image_path = None
        self.current_pixmap = None
        self.original_size = None
        self.thumbnail_labels = []
        self.base_metadata_text = "" # Initialize metadata text
        
        # --- STATE TRACKING ---
        self.current_folder = None 
        self.category_history = [] 
        self.last_selected_path = None
        
        # User Manual Link Configuration
        # The link has been updated as requested:
        self.user_manual_url = "file:///C:/Users/Lenovo/Desktop/User%20Manual.pdf" 
        
        # ================= NEW: Toolbar Setup =================
        self._create_toolbar()

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
        self.metadata_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

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

        # --- Load progress after UI is initialized ---
        self._load_progress()
        if self.current_folder:
             # Update category label on resume
             self.category_label.setText(f"Loaded {len(self.category_history)} previous categorizations.")

    # ================= Toolbar Methods =================

    def _create_toolbar(self):
        toolbar = self.addToolBar("Main Toolbar")
        
        # 1. Save Button/Action
        save_action = QAction("Save", self)
        save_action.setStatusTip("Save current categorization progress, folder, and selected image")
        save_action.triggered.connect(self.save_progress)
        toolbar.addAction(save_action)

        # 2. Undo Button/Action 
        undo_action = QAction("Undo", self)
        undo_action.setStatusTip("Revert the last categorization action")
        undo_action.triggered.connect(self.undo_action)
        toolbar.addAction(undo_action)
        
        # 3. Help Button (with Dropdown Menu)
        help_action = QAction("Help", self)
        help_action.setStatusTip("Show help options")
        
        help_menu = QMenu(self)
        
        manual_action = QAction("User Manual", self)
        manual_action.triggered.connect(self.open_user_manual)
        help_menu.addAction(manual_action)
        
        help_action.setMenu(help_menu)
        toolbar.addAction(help_action)

    def _load_progress(self):
        """Loads the history stack, folder path, and last selected image from the save file on startup."""
        if os.path.exists(self.SAVE_FILE):
            try:
                with open(self.SAVE_FILE, 'r') as f:
                    state = json.load(f)
                    self.category_history = state.get("history", [])
                    self.current_folder = state.get("folder", None)
                    self.last_selected_path = state.get("selected_image", None)
                
                print(f"Loaded state with {len(self.category_history)} past actions.")
                
                # Automatically load the folder if a path was saved
                if self.current_folder and os.path.isdir(self.current_folder):
                    self._load_folder_on_resume(self.current_folder, self.last_selected_path)
                    QMessageBox.information(self, "Resume Session", f"Resumed session from folder:\n{self.current_folder}")
                elif self.current_folder:
                     print(f"Saved folder path '{self.current_folder}' not found. Starting fresh.")

            except Exception as e:
                print(f"Error loading state: {e}")
                QMessageBox.warning(self, "Load Error", f"Could not load previous progress: {e}")
                self.category_history = []
                self.current_folder = None
                self.last_selected_path = None
        else:
            print("No saved state found.")

    def save_progress(self):
        """Saves the history stack, current folder, and last selected image to a file."""
        if not self.current_folder:
            print("Save Error: Cannot save: No folder has been uploaded yet.")
            QMessageBox.warning(self, "Save Error", "Cannot save: No folder has been uploaded yet.")
            return

        state = {
            "folder": self.current_folder,
            "history": self.category_history,
            "selected_image": self.current_image_path
        }
        
        try:
            with open(self.SAVE_FILE, 'w') as f:
                json.dump(state, f, indent=4)
            print(f"Progress saved! {len(self.category_history)} actions recorded.")
            QMessageBox.information(self, "Save Success", f"Progress saved successfully ({len(self.category_history)} actions).")
        except Exception as e:
            print(f"Error saving progress: {e}")
            QMessageBox.critical(self, "Save Error", f"Could not save progress: {e}")

    def undo_action(self):
        """Reverts the last categorization action AND clears the current image selection."""
        if not self.category_history:
            self.category_label.setText("Category: Nothing to undo.")
            print("Undo failed: History is empty.")
            
            # If history is empty, but an image is selected, still clear it.
            if self.current_image_path:
                 self.unselect_current_image()
            return

        last_action = self.category_history.pop()
        
        # 1. Update the display
        self.category_label.setText(
            f"Category: UNDONE! Last action on {os.path.basename(last_action['path'])} was '{last_action['category']}'."
        )
        print(f"Undid action: {last_action}")

        # 2. Clear the image selection
        self.unselect_current_image()

    def unselect_current_image(self):
        """Helper to clear the display state when undoing or changing folders."""
        
        # Reset current selection variables
        self.current_image_path = None
        self.current_pixmap = None
        self.original_size = None
        self.last_selected_path = None # Clear the last selected path as well

        # Reset main image view
        self.image_label.setPixmap(QPixmap()) # Clears the image
        self.image_label.setText("Click a thumbnail to show here")
        
        # Reset metadata
        self.metadata_label.setText("Metadata will appear here")
        
        # Reset thumbnail highlights
        for thumb in self.thumbnail_labels:
            thumb.setStyleSheet("border: 1px solid gray; margin: 2px;")


    def open_user_manual(self):
        """Opens the user manual link."""
        # The URL check has been updated to reflect the new file path
        if self.user_manual_url:
            import webbrowser
            webbrowser.open(self.user_manual_url)
            print(f"--- Opening User Manual: {self.user_manual_url} ---")
        else:
            print("--- User Manual link not configured. Please update self.user_manual_url. ---")
            
    # ================= Upload Folder =================
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
             # Clear current selection before loading new folder
             self.unselect_current_image() 
             self.current_folder = folder # Store the new folder
             self._load_thumbnails(folder)

    def _load_folder_on_resume(self, folder, selected_path=None):
        """Helper to load the folder and potentially select an image on resume."""
        self.unselect_current_image()
        self._load_thumbnails(folder)
        
        # Automatically select the last image if a valid path was saved
        if selected_path and os.path.exists(selected_path):
            # Use on_thumbnail_clicked to simulate the user action
            self.on_thumbnail_clicked(selected_path)


    def _load_thumbnails(self, folder):
        """Helper to load the thumbnails for a given folder."""
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
                # Try to load and scale the image
                try:
                    pixmap = QPixmap(path).scaledToWidth(100, Qt.SmoothTransformation)
                except Exception as e:
                    print(f"Error loading image {fname}: {e}")
                    continue

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
        # Store the path of the newly selected image
        self.last_selected_path = path

        # Reset all borders
        for thumb in self.thumbnail_labels:
            thumb.setStyleSheet("border: 1px solid gray; margin: 2px;")
        
        # Highlight selected one
        # Use a list comprehension to find the clicked label based on path
        clicked_label = None
        try:
            clicked_label = next(thumb for thumb in self.thumbnail_labels if thumb.path == path)
            clicked_label.setStyleSheet("border: 2px solid blue; margin: 2px;")
        except StopIteration:
            # Should not happen if path is correct and exists in thumbnails
            pass 

        # Load image
        self.load_image(path)

    # ================= Load Image =================
    # (Remains unchanged)
    def load_image(self, path):
        self.current_image_path = path
        self.current_pixmap = QPixmap(path)

        # Note: PIL/Pillow must be installed for the following to work
        try:
            img = Image.open(path)
            self.original_size = img.size
        except Exception as e:
            # Handle potential PIL error (e.g., corrupted file)
            print(f"Error opening image with PIL: {e}")
            self.original_size = ("N/A", "N/A")

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
    # (Remains unchanged)
    def show_metadata(self, fname):
        # Note: PIL/Pillow must be installed for the following to work
        try:
            img = Image.open(fname)
            img_format = img.format
            img_mode = img.mode
            img_size = f"{img.size[0]} x {img.size[1]} px"
        except Exception:
            img_format = "Unknown"
            img_mode = "Unknown"
            img_size = "N/A"

        # File size calculation
        try:
            file_size_bytes = os.path.getsize(fname)
            if file_size_bytes < 1024:
                file_size_str = f"{file_size_bytes} B"
            elif file_size_bytes < 1024 * 1024:
                file_size_str = f"{file_size_bytes/1024:.2f} KB"
            else:
                file_size_str = f"{file_size_bytes/(1024*1024):.2f} MB"
        except Exception:
            file_size_str = "N/A"

        self.base_metadata_text = (
            f"File: {os.path.basename(fname)}\n"
            f"Format: {img_format}\n"
            f"Original Size: {img_size}\n"
            f"Mode: {img_mode}\n"
            f"File Size: {file_size_str}"
        )
        self.update_metadata_display()

    def update_metadata_display(self):
        if self.current_pixmap:
            displayed_w = self.image_label.pixmap().width() if self.image_label.pixmap() else 0
            displayed_h = self.image_label.pixmap().height() if self.image_label.pixmap() else 0
            displayed_size = f"Displayed Pixmap Size: {displayed_w} x {displayed_h} px"
            
            # Use label's current size as a fallback/context
            label_w = self.image_label.width()
            label_h = self.image_label.height()
            label_size = f"Label Area Size: {label_w} x {label_h} px"
            
            self.metadata_label.setText(
                self.base_metadata_text + "\n" + displayed_size + "\n" + label_size
            )

    # ================= Categorization =================
    def categorize_image(self):
        if not self.current_image_path:
            self.category_label.setText("No image selected!")
            return

        import random
        category = random.choice(["Repair", "Accept"])
        
        # 1. Update the display
        self.category_label.setText(f"Result: {category}")

        # 2. Record the action for history/saving (NEW)
        action = {
            "path": self.current_image_path,
            "category": category
        }
        self.category_history.append(action)
        print(f"Action recorded: {action}")


def main():
    # Make sure PIL/Pillow is installed before running!
    app = QApplication(sys.argv)
    window = DRDOGUI()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
