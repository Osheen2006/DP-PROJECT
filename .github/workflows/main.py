import sys
import os
import json 
import random
from PyQt5.QtGui import QPainter, QColor

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QWidget, QGridLayout, QScrollArea,
    QAction, QMenu, QMessageBox 
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal
from report_generator import generate_report, open_reports_folder

from models import SegmentationModels


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
    SAVE_FILE = "app_state.json"
    def run_all_models(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Select an image first.")
            return
    
        try:
            from models import run_all_models
            result = run_all_models(self.current_image_path)
    
            # Show in output box
            self.output_label.setText(str(result))
    
            print("Model results:", result)
        except Exception as e:
            QMessageBox.critical(self, "Model Error", str(e))
            print("Error running models:", e)

    def handle_generate_report(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Select an image first.")
            return

        generate_report(self, self.current_image_path, self.metadata_label.text())



    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basic Frontend")
        self.setGeometry(200, 200, 1700, 900)

        self.current_image_path = None
        self.current_pixmap = None
        self.original_size = None
        self.thumbnail_labels = []
        self.base_metadata_text = "" # Initialize metadata text
        
        # --- STATE TRACKING ---
        self.current_folder = None 
        self.category_history = [] 
        self.last_selected_path = None
        
       
        self.user_manual_url = "file:///C:/Users/Lenovo/Desktop/User%20Manual.pdf" 
        
        
        self._create_toolbar()

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

                # Top-Left: Selected Image + Detect Defect Button
        selected_img_box = QVBoxLayout()
        
        self.image_label = QLabel("Click a thumbnail to show here")
        self.image_label.setStyleSheet("border: 1px solid gray; min-height: 200px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        
        selected_img_box.addWidget(self.image_label)
        
        # === DETECT DEFECT BUTTON HERE ===
        self.model_btn = QPushButton("Detect Defect")
        self.model_btn.clicked.connect(self.run_all_models)   # <-- you will define this method
        selected_img_box.addWidget(self.model_btn)
        
        # Place this compound widget into the grid
        img_section = QWidget()
        img_section.setLayout(selected_img_box)
        self.layout.addWidget(img_section, 0, 1)


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

        self.model_btn = QPushButton("Detect Defect")

        self.layout.addWidget(self.model_btn, 2, 1)

                # Bottom-Left: Categorize
        self.categorize_btn = QPushButton("Categorize (Repair / Accept)")
        self.categorize_btn.clicked.connect(self.categorize_image)
        self.generate_report_btn = QPushButton("Generate Report")
        self.generate_report_btn.clicked.connect(self.handle_generate_report)
        self.layout.addWidget(self.generate_report_btn, 2, 1)

        self.open_reports_btn = QPushButton("Open Reports")
        self.open_reports_btn.clicked.connect(open_reports_folder)
        self.layout.addWidget(self.open_reports_btn, 2, 2)

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

        # ================= New Rightmost Column: Bounding Boxes (column 3) =================
        # This column is intentionally thin and contains a scrollable list of detected boxes.
        self.bb_title = QLabel("Bounding Boxes")
        self.bb_title.setAlignment(Qt.AlignCenter)
        self.bb_title.setStyleSheet("font-weight: bold; padding: 4px;")

        self.bb_scroll = QScrollArea()
        self.bb_scroll.setWidgetResizable(True)
        self.bb_content = QWidget()
        self.bb_layout = QVBoxLayout(self.bb_content)
        self.bb_layout.setAlignment(Qt.AlignTop)
        self.bb_content.setLayout(self.bb_layout)
        self.bb_scroll.setWidget(self.bb_content)

        vbox_bb = QVBoxLayout()
        vbox_bb.addWidget(self.bb_title)
        vbox_bb.addWidget(self.bb_scroll)

        section_bb = QWidget()
        section_bb.setLayout(vbox_bb)
        # place it at column 3 spanning both rows
        self.layout.addWidget(section_bb, 0 , 3, 2,1)
        # Layout proportions
        self.layout.setColumnStretch(0, 1)  # thin left
        self.layout.setColumnStretch(1, 3)
        self.layout.setColumnStretch(2, 2)
        self.layout.setColumnStretch(3, 1)
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
# ================= Bounding Box UI Helpers =================
    def _clear_bb_list(self):
        for i in reversed(range(self.bb_layout.count())):
            widget = self.bb_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

    def _add_bb_item(self, class_name, coords, color_hex):
        # Each item shows a small color swatch + class label + coordinates text
        item_widget = QWidget()
        h = QVBoxLayout()
        h.setContentsMargins(6, 4, 6, 4)
        h.setSpacing(8)

        # color swatch
        swatch = QLabel()
        swatch.setFixedSize(16, 16)
        swatch.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #222;")
        h.addWidget(swatch)

        # class name (colored text)
        class_lbl = QLabel(class_name)
        class_lbl.setStyleSheet(f"font-weight: bold; color: {color_hex};")
        h.addWidget(class_lbl)

        # coordinates
        coords_lbl = QLabel(f"{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}")
        coords_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        h.addWidget(coords_lbl)

        # spacer so items compress nicely in the thin column
        h.addStretch()
        item_widget.setLayout(h)
        self.bb_layout.addWidget(item_widget)

    def display_bounding_boxes(self, boxes):
        """
        Accepts a list of boxes where each box is:
        {"class": str, "coords": [x1,y1,x2,y2], "color": "#RRGGBB"}
        This updates the rightmost column UI.
        """
        # store
        self.current_bboxes = boxes or []
        self._clear_bb_list()

        if not boxes:
            empty_lbl = QLabel("No bounding boxes detected.")
            empty_lbl.setStyleSheet("color: gray; padding: 6px;")
            self.bb_layout.addWidget(empty_lbl)
            return

        for box in boxes:
            cls = box.get("class", "unk")
            coords = box.get("coords", ["N/A", "N/A", "N/A", "N/A"])
            color = box.get("color", "#AAAAAA")
            self._add_bb_item(cls, coords, color)

    # ================= Categorization / Detection =================
    def categorize_image(self):
        if not self.current_image_path:
            self.category_label.setText("No image selected!")
            return

        # Random categorize as before
        category = random.choice(["Repair", "Accept"])
        self.category_label.setText(f"Result: {category}")

        # Record the action
        action = {
            "path": self.current_image_path,
            "category": category
        }
        self.category_history.append(action)
        print(f"Action recorded: {action}")

        # After categorization, attempt to detect bboxes using a model if available,
        # otherwise generate a simulated example for demonstration.
        boxes = []
        try:
            # Try to import a user-provided detection function.
            # Expected function signature: detect_bboxes(image_path) -> list of dicts
            # with keys 'class', 'coords', 'color' (color optional)
            from model import detect_bboxes  # user should provide this module if available
            boxes = detect_bboxes(self.current_image_path)
            print("Detected boxes via model.detect_bboxes()")
        except Exception as e:
            # If the user model isn't present, make up a few example boxes so UI is usable.
            print(f"No external model detected or model failed ({e}). Using simulated boxes.")
            # Simulate up to 3 boxes with random-ish coords scaled to original image size (if known)
            w, h = (self.original_size if isinstance(self.original_size, tuple) else (640, 480))
            if not (isinstance(w, int) and isinstance(h, int)):
                w, h = 640, 480
            classes = ["scratch", "dent", "misalign"]
            palette = ["#FF4D4D", "#FFA500", "#4DA6FF"]
            for i in range(random.randint(1, 3)):
                x1 = random.randint(0, int(w * 0.6))
                y1 = random.randint(0, int(h * 0.6))
                x2 = x1 + random.randint(20, int(w * 0.3))
                y2 = y1 + random.randint(20, int(h * 0.3))
                boxes.append({
                    "class": classes[i % len(classes)],
                    "coords": [x1, y1, min(x2, w), min(y2, h)],
                    "color": palette[i % len(palette)]
                })

        # Update the bounding boxes column UI
        self.display_bounding_boxes(boxes)

        # Optionally draw boxes on the displayed preview (scaled) so user sees approximate positions
        self._overlay_boxes_on_preview(boxes)

    def _overlay_boxes_on_preview(self, boxes):
        """
        Draws bounding boxes on a copy of the current pixmap to show approximate positions.
        This uses the scaled pixmap currently shown in the image_label, so coordinates are scaled.
        """
        if not self.current_pixmap or not boxes:
            return

        pixmap = self.current_pixmap
        # create a copy to paint on
        painted = QPixmap(pixmap)
        painter = QPainter(painted)
        painter.setRenderHint(QPainter.Antialiasing)

        # Determine scale from original image size to displayed pixmap size
        try:
            orig_w, orig_h = self.original_size
            disp_pixmap = self.image_label.pixmap()
            if disp_pixmap:
                disp_w = disp_pixmap.width()
                disp_h = disp_pixmap.height()
            else:
                disp_w = painted.width()
                disp_h = painted.height()

            sx = disp_w / max(orig_w, 1)
            sy = disp_h / max(orig_h, 1)
        except Exception:
            sx = sy = 1.0

        for box in boxes:
            coords = box.get("coords", [0, 0, 0, 0])
            color_hex = box.get("color", "#FF0000")
            try:
                color = QColor(color_hex)
            except Exception:
                color = QColor("#FF0000")
            painter.setPen(color)
            # scale coords
            x1 = int(coords[0] * sx)
            y1 = int(coords[1] * sy)
            x2 = int(coords[2] * sx)
            y2 = int(coords[3] * sy)
            painter.drawRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        painter.end()

        # Resize the painted pixmap to label keeping aspect ratio
        scaled_painted = painted.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_painted)

    def closeEvent(self, event):
        # Ensure we save state on close
        try:
            self.save_progress()
        except Exception:
            pass
        super().closeEvent(event)

def SegentationModels(self):
    if not self.current_image_path:
        QMessageBox.warning(self, "Error", "No image selected!")
        return

    try:
        # Choose one of your models
        model_choice = "segformer"  # or "unet" / "unetpp"
        result = self.seg_models.segment(self.current_image_path, model_choice)

        # Save and display result
        output_path = os.path.join(self.current_folder, "segmented_output.png")
        result.save(output_path)

        self.output_label.setPixmap(QPixmap(output_path).scaledToWidth(400, Qt.SmoothTransformation))
        self.category_label.setText(f"Segmentation done using {model_choice.upper()}")
        print(f"Saved output to {output_path}")
    except Exception as e:
        QMessageBox.critical(self, "Segmentation Error", str(e))
        print("Error:", e)

def main():
    # Make sure PIL/Pillow is installed before running!
    app = QApplication(sys.argv)
    window = DRDOGUI()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
