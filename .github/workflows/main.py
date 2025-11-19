import sys
import os
import json
import random
import webbrowser
from PyQt5.QtGui import QPainter, QColor, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QApplication, QMainWindow, QPushButton, QLabel,
    QSizePolicy, QFileDialog, QVBoxLayout, QWidget, QGridLayout,
    QScrollArea, QAction, QMenu, QMessageBox, QHBoxLayout, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PIL import Image, ImageFilter
import numpy as np

# Assuming these modules exist in the same environment
from report_generator import generate_report, open_reports_folder
from models import SegmentationModels, DetectionModels, detect_bboxes


# Custom QLabel that emits a signal when clicked
class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)  # will emit the image path

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.path)


class GUI(QMainWindow):
    SAVE_FILE = "app_state.json"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weld Vision: Segmentation-Detection")
        self.setGeometry(200, 200, 1700, 900)

        # --- Presentation: Apply modern style ---
        self._set_style_sheet(QApplication.instance())
        # ----------------------------------------

        # Model instances
        self.seg_models = SegmentationModels(device="cpu")
        self.det_models = DetectionModels(device="cpu")

        # state
        self.current_image_path = None
        self.current_pixmap = None
        self.original_size = None
        self.thumbnail_labels = []
        self.base_metadata_text = ""
        self.current_folder = None
        self.category_history = []
        self.last_selected_path = None
        self.user_manual_url = "file:///C:/Users/Lenovo/Desktop/WeldVision_User_Manual.pdf"

        # New/Updated state variables for tracking results and reporting
        self.current_bboxes = []
        self.detection_output_path = None  # Path to saved image with detection boxes
        self.segmentation_output_path = None  # Path to saved segmentation (COLOURED) image

        self._create_toolbar()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)

        # ----------------------------------------------------
        # 1. Left column: Image Selection (Upload + Thumbnails)
        # ----------------------------------------------------

        self.upload_btn = QPushButton("📂 Upload Folder")
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

        group_left = QGroupBox("Image Selection")
        group_left.setLayout(vbox_left)
        self.layout.addWidget(group_left, 0, 0, 3, 1)

        # ----------------------------------------------------
        # 2. Center (Top): Image Preview & Segmentation Control
        # ----------------------------------------------------

        selected_img_box = QVBoxLayout()
        self.image_label = QLabel("Click a thumbnail to show here")
        self.image_label.setObjectName("ImageLabel")
        self.image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 200px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)

        # Segmentation Model Group
        seg_model_group = QWidget()
        seg_hlayout = QGridLayout(seg_model_group)

        seg_label = QLabel("Segmentation Model:")
        self.seg_model_dropdown = QComboBox()
        self.seg_model_dropdown.addItems(list(self.seg_models.model_paths.keys()))
        self.seg_model_btn = QPushButton("🏃 Run Segmentation")
        self.seg_model_btn.clicked.connect(self.run_segmentation)

        seg_hlayout.addWidget(seg_label, 0, 0)
        seg_hlayout.addWidget(self.seg_model_dropdown, 0, 1)
        seg_hlayout.addWidget(self.seg_model_btn, 0, 2)
        seg_hlayout.setColumnStretch(1, 1)

        selected_img_box.addWidget(self.image_label)
        selected_img_box.addWidget(seg_model_group)

        group_center_top = QGroupBox("Image Preview & Segmentation")
        group_center_top.setLayout(selected_img_box)
        self.layout.addWidget(group_center_top, 0, 1, 1, 1)

        # ----------------------------------------------------
        # 3. Top-Right: Segmentation Output Mask
        # ----------------------------------------------------

        self.proc_title = QLabel("🖼️ Segmentation Output Mask")
        self.proc_title.setStyleSheet("font-weight: bold; padding: 4px; color: #0078D4;")
        
        self.output_label = QLabel("Segmentation mask will appear here")
        self.output_label.setObjectName("OutputLabel")
        # --- MODIFIED: min-height is set in _set_style_sheet/load_image/unselect_current_image for visibility ---
        self.output_label.setStyleSheet("border: 1px solid #ccc; min-height: 150px; background-color: #fff;")
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setScaledContents(True)

        vbox_proc = QVBoxLayout()
        vbox_proc.addWidget(self.proc_title)
        vbox_proc.addWidget(self.output_label)
        # Removed vbox_proc.addStretch() to allow the label to naturally fill space
        
        group_top_right = QGroupBox("Process Output")
        group_top_right.setLayout(vbox_proc)
        # --- MODIFIED: Set vertical size policy back to Preferred to allow growth ---
        group_top_right.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred) 
        self.layout.addWidget(group_top_right, 0, 2)

        # ----------------------------------------------------
        # 4. Center (Middle): Detection & Categorization
        # ----------------------------------------------------

        det_model_group = QWidget()
        det_hlayout = QGridLayout(det_model_group)

        det_label = QLabel("Detection Model:")
        self.det_model_dropdown = QComboBox()
        self.det_model_dropdown.addItems(list(self.det_models.model_paths.keys()))
        self.categorize_btn = QPushButton("🎯 Run Detection / Categorize")
        self.categorize_btn.clicked.connect(self.run_detection)

        det_hlayout.addWidget(det_label, 0, 0)
        det_hlayout.addWidget(self.det_model_dropdown, 0, 1)
        det_hlayout.addWidget(self.categorize_btn, 0, 2)
        det_hlayout.setColumnStretch(1, 1)

        self.category_label = QLabel("Detection / Categorization result will appear here")
        self.category_label.setStyleSheet(
            "border: 1px solid #0078D4; background-color: #E6F0F8; padding: 8px; font-weight: bold; min-height: 30px;")
        self.category_label.setAlignment(Qt.AlignCenter)

        vbox_cat = QVBoxLayout()
        vbox_cat.addWidget(det_model_group)
        vbox_cat.addWidget(self.category_label)

        group_center_mid = QGroupBox("Detection & Categorization")
        group_center_mid.setLayout(vbox_cat)
        self.layout.addWidget(group_center_mid, 1, 1)

        # ----------------------------------------------------
        # 5. Bottom Right: Metadata
        # ----------------------------------------------------

        self.metadata_label = QLabel("Metadata will appear here")
        self.metadata_label.setStyleSheet("border: 1px solid #ccc; background-color: #fff; padding: 8px;")
        self.metadata_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        vbox_meta = QVBoxLayout()
        vbox_meta.addWidget(self.metadata_label)

        group_meta = QGroupBox("Image Metadata")
        group_meta.setLayout(vbox_meta)
        self.layout.addWidget(group_meta, 1, 2)

        # ----------------------------------------------------
        # 6. Center (Bottom): Report Buttons
        # ----------------------------------------------------

        report_hlayout = QHBoxLayout()
        self.generate_report_btn = QPushButton("📄 Generate Report")
        self.generate_report_btn.clicked.connect(self.handle_generate_report)
        self.open_reports_btn = QPushButton("📂 Open Reports Folder")
        self.open_reports_btn.clicked.connect(open_reports_folder)

        report_hlayout.addWidget(self.generate_report_btn)
        report_hlayout.addWidget(self.open_reports_btn)

        group_report = QGroupBox("Reporting")
        group_report.setLayout(report_hlayout)

        self.layout.addWidget(group_report, 2, 1, 1, 2)

        # ----------------------------------------------------
        # 7. Rightmost column: Bounding boxes list
        # ----------------------------------------------------

        self.bb_title = QLabel("Detected Defects List")
        self.bb_title.setAlignment(Qt.AlignCenter)
        self.bb_title.setStyleSheet("font-weight: bold; padding: 4px; color: #CC0000;")

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

        group_bb = QGroupBox("Bounding Box Details")
        group_bb.setLayout(vbox_bb)
        self.layout.addWidget(group_bb, 0, 3, 3, 1)

        # ----------------------------------------------------
        # 8. Layout Proportions (Modified for larger images, smaller detection/wider C2)
        # ----------------------------------------------------

        # Column stretches: 1 (L), 3 (C1), 3 (C2-Wider), 1 (R)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 3)
        self.layout.setColumnStretch(2, 3)  
        self.layout.setColumnStretch(3, 1)
        
        # Row stretches: 6 (Images-Larger), 2 (Detection-Smaller), 1 (Report)
        self.layout.setRowStretch(0, 6) 
        self.layout.setRowStretch(1, 2)
        self.layout.setRowStretch(2, 1)

        # load saved progress (if any)
        self._load_progress()
        QApplication.processEvents()
        if self.current_folder:
            self.category_label.setText(f"Loaded {len(self.category_history)} previous categorizations.")
        if self.current_image_path:
            self.resize_image_to_label()

    # ----------------------------------------------------
    # Styling and Helper Methods
    # ----------------------------------------------------
    def _set_style_sheet(self, app):
        """Sets a modern, professional CSS style for the application."""
        app.setStyle("Fusion")

        style_sheet = """
        QMainWindow {
            background-color: #f0f0f0; 
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            margin-top: 1ex; 
            padding: 5px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left; 
            padding: 0 3px;
            color: #333;
        }
        QPushButton {
            background-color: #0078D4; 
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
            min-height: 25px;
        }
        QPushButton:hover {
            background-color: #005A9E;
        }
        QPushButton:pressed {
            background-color: #00457A;
        }
        QComboBox, QLineEdit {
            border: 1px solid #ccc;
            padding: 5px;
            border-radius: 4px;
            min-height: 25px;
            background-color: white;
        }
        QLabel#ImageLabel {
            border: 2px solid #aaa;
            background-color: #fff;
            min-height: 200px; /* Keep this for ImageLabel as a baseline */
        }
        QLabel#OutputLabel {
            border: 1px solid #ccc; 
            min-height: 150px; /* MODIFIED: Set a new minimum height */
            background-color: #fff;
        }
        QLabel {
            padding: 2px;
        }
        QScrollArea {
            border: 1px solid #ccc;
            background-color: #fff;
        }
        """
        self.setStyleSheet(style_sheet)

    # ----------------------------------------------------
    # Image/Mask helper: multiply original by mask (per-pixel)
    # ----------------------------------------------------
    def multiply_image_by_mask(self, original_image_path: str, mask_pil: Image.Image, blur_radius: float = 0.0) -> Image.Image:
        """
        Cross-multiply the original RGB image by the segmentation mask.
        - original_image_path: path to the RGB input image
        - mask_pil: PIL Image returned by seg.segment(...) (could be RGB or L)
        - blur_radius: if >0, will apply a small gaussian blur to soften hard binary masks
        Returns: PIL RGB image = original * mask (soft multiplication)
        """
        try:
            orig = Image.open(original_image_path).convert("RGB")
        except Exception:
            # fallback: create a blank image if original can't be opened
            orig = Image.new("RGB", (640, 480), (0, 0, 0))

        # Ensure mask single-channel and same size
        mask = mask_pil.convert("L")
        if blur_radius and blur_radius > 0.0:
            try:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            except Exception:
                pass

        if mask.size != orig.size:
            mask = mask.resize(orig.size, resample=Image.NEAREST)

        a = np.asarray(orig).astype(np.float32) / 255.0   # H x W x 3
        m = np.asarray(mask).astype(np.float32) / 255.0  # H x W

        m3 = np.expand_dims(m, axis=2)                    # H x W x 1
        out = (a * m3)
        out_img = Image.fromarray((out * 255.0).clip(0, 255).astype(np.uint8))
        return out_img

    # ----------------------------------------------------
    # Original Application Methods (Functionality)
    # ----------------------------------------------------

    # Toolbar
    def _create_toolbar(self):
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setStyleSheet("QToolBar { background-color: #e0e0e0; }")
        save_action = QAction("💾 Save", self)
        save_action.setStatusTip("Save current categorization progress, folder, and selected image")
        save_action.triggered.connect(self.save_progress)
        toolbar.addAction(save_action)

        undo_action = QAction("↩️ Undo", self)
        undo_action.setStatusTip("Revert the last categorization action")
        undo_action.triggered.connect(self.undo_action)
        toolbar.addAction(undo_action)

        help_action = QAction("❓ Help", self)
        help_action.setStatusTip("Show help options")
        help_menu = QMenu(self)
        manual_action = QAction("User Manual", self)
        manual_action.triggered.connect(self.open_user_manual)
        help_menu.addAction(manual_action)
        help_action.setMenu(help_menu)
        toolbar.addAction(help_action)

    def _load_progress(self):
        if os.path.exists(self.SAVE_FILE):
            try:
                with open(self.SAVE_FILE, 'r') as f:
                    state = json.load(f)
                    self.category_history = state.get("history", [])
                    self.current_folder = state.get("folder", None)
                    self.last_selected_path = state.get("selected_image", None)
                print(f"Loaded state with {len(self.category_history)} past actions.")
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
        if not self.category_history:
            self.category_label.setText("Category: Nothing to undo.")
            print("Undo failed: History is empty.")
            if self.current_image_path:
                self.unselect_current_image()
            return
        last_action = self.category_history.pop()
        self.category_label.setText(
            f"Category: UNDONE! Last action on {os.path.basename(last_action['path'])} was '{last_action['category']}'.")
        print(f"Undid action: {last_action}")
        if self.current_image_path:
            self.load_image(self.current_image_path, clear_bb=True)
        else:
            self.unselect_current_image()

    def unselect_current_image(self):
        # Reset state variables related to output paths
        self.current_image_path = None
        self.current_pixmap = None
        self.original_size = None
        self.last_selected_path = None
        self.detection_output_path = None
        self.segmentation_output_path = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("Click a thumbnail to show here")
        self.image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 200px;")
        self.metadata_label.setText("Metadata will appear here")
        self.output_label.setPixmap(QPixmap())
        self.output_label.setText("Segmentation mask will appear here")
        # Ensure correct stylesheet is used after unselection
        self.output_label.setStyleSheet("border: 1px solid #ccc; min-height: 150px; background-color: #fff;")
        self._clear_bb_list()
        for thumb in self.thumbnail_labels:
            thumb.setStyleSheet("border: 1px solid #ccc; margin: 2px;")

    def open_user_manual(self):
        if self.user_manual_url:
            webbrowser.open(self.user_manual_url)
            print(f"--- Opening User Manual: {self.user_manual_url} ---")
        else:
            print("--- User Manual link not configured. ---")

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.unselect_current_image()
            self.current_folder = folder
            self._load_thumbnails(folder)

    def _load_folder_on_resume(self, folder, selected_path=None):
        self.unselect_current_image()
        self._load_thumbnails(folder)
        if selected_path and os.path.exists(selected_path):
            self.on_thumbnail_clicked(selected_path)

    def _load_thumbnails(self, folder):
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.thumbnail_labels.clear()
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(folder, fname)
                try:
                    pixmap = QPixmap(path).scaledToWidth(100, Qt.SmoothTransformation)
                except Exception as e:
                    print(f"Error loading image {fname}: {e}")
                    continue
                thumb_label = ClickableLabel(path)
                thumb_label.setPixmap(pixmap)
                thumb_label.setStyleSheet("border: 1px solid #ccc; margin: 2px;")
                thumb_label.setCursor(Qt.PointingHandCursor)
                thumb_label.clicked.connect(self.on_thumbnail_clicked)
                self.scroll_layout.addWidget(thumb_label)
                self.thumbnail_labels.append(thumb_label)

    def on_thumbnail_clicked(self, path):
        self.last_selected_path = path
        for thumb in self.thumbnail_labels:
            thumb.setStyleSheet("border: 1px solid #ccc; margin: 2px;")
        try:
            clicked_label = next(thumb for thumb in self.thumbnail_labels if thumb.path == path)
            clicked_label.setStyleSheet("border: 2px solid #0078D4; margin: 2px;")
        except StopIteration:
            pass
        self.load_image(path)

    def load_image(self, path, clear_bb=True):
        # Reset output paths whenever a new image is loaded
        self.detection_output_path = None
        self.segmentation_output_path = None

        self.current_image_path = path
        self.current_pixmap = QPixmap(path)
        try:
            img = Image.open(path)
            self.original_size = img.size
        except Exception as e:
            print(f"Error opening image with PIL: {e}")
            self.original_size = ("N/A", "N/A")

        self.resize_image_to_label()
        self.show_metadata(path)
        self.output_label.setPixmap(QPixmap())
        self.output_label.setText("Segmentation mask will appear here")
        self.output_label.setStyleSheet("border: 1px solid #ccc; min-height: 150px; background-color: #fff;")
        if clear_bb:
            self._clear_bb_list()
            self.category_label.setText("Detection / Categorization result will appear here")

    def resize_image_to_label(self):
        if self.current_pixmap:
            scaled_pixmap = self.current_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio,
                                                       Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setStyleSheet("border: 2px solid #aaa; background-color: #fff;")
            self.update_metadata_display()
            # Also update output label if an image is loaded there
            if self.segmentation_output_path and os.path.exists(self.segmentation_output_path):
                self.output_label.setPixmap(QPixmap(self.segmentation_output_path).scaled(
                    self.output_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def resizeEvent(self, event):
        self.resize_image_to_label()
        super().resizeEvent(event)

    def show_metadata(self, fname):
        try:
            img = Image.open(fname)
            img_format = img.format
            img_mode = img.mode
            img_size = f"{img.size[0]} x {img.size[1]} px"
        except Exception:
            img_format = "Unknown"
            img_mode = "Unknown"
            img_size = "N/A"
        try:
            file_size_bytes = os.path.getsize(fname)
            if file_size_bytes < 1024:
                file_size_str = f"{file_size_bytes} B"
            elif file_size_bytes < 1024 * 1024:
                file_size_str = f"{file_size_bytes / 1024:.2f} KB"
            else:
                file_size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"
        except Exception:
            file_size_str = "N/A"
        self.base_metadata_text = (
            f"File: **{os.path.basename(fname)}**\n"
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
            label_w = self.image_label.width()
            label_h = self.image_label.height()
            label_size = f"Label Area Size: {label_w} x {label_h} px"
            self.metadata_label.setText(self.base_metadata_text + "\n" + displayed_size + "\n" + label_size)

    def run_detection(self):
        if not self.current_image_path:
            self.category_label.setText("No image selected!")
            QMessageBox.warning(self, "No Image", "Select an image first to run detection.")
            return

        model_name = self.det_model_dropdown.currentText()
        self.category_label.setText(f"Running detection using **{model_name.upper()}**...")
        QApplication.processEvents()

        try:
            boxes = detect_bboxes(self.current_image_path, model_name)

            if boxes and boxes[0].get("class") != "simulated_defect":
                category = "Repair (Defect Detected)"
            elif boxes and boxes[0].get("class") == "simulated_defect":
                category = "Review (Model Error/Simulated)"
            else:
                category = "Accept (No Defect Found)"

            self.category_label.setText(f"Result: **{category}**")
            action = {"path": self.current_image_path, "category": category, "model": model_name}
            self.category_history.append(action)
            print(f"Action recorded: {action}")

        except Exception as e:
            QMessageBox.critical(self, "Detection Error", str(e))
            self.category_label.setText(f"Detection Failed! Error: {e}")
            print("detect_bboxes failed:", e)
            boxes = []

        self.display_bounding_boxes(boxes)

        # --- Run Overlay & Save the visualization for reporting ---
        if self.current_image_path and self.current_folder:
            det_output_name = f"{os.path.splitext(os.path.basename(self.current_image_path))[0]}_detection_overlay.png"
            self.detection_output_path = os.path.join(self.current_folder, det_output_name)

            painted_pixmap = self._overlay_boxes_on_preview(boxes)
            if painted_pixmap:
                painted_pixmap.save(self.detection_output_path)
            else:
                self.detection_output_path = None
        # ---------------------------------------------------------------

    def _clear_bb_list(self):
        for i in reversed(range(self.bb_layout.count())):
            widget = self.bb_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.current_bboxes = []

    def _add_bb_item(self, class_name, coords, color_hex, confidence="N/A"):
        item_widget = QWidget()
        v = QVBoxLayout()
        v.setContentsMargins(6, 4, 6, 4)
        v.setSpacing(4)

        h_class = QWidget()
        h_layout = QHBoxLayout(h_class)
        h_layout.setContentsMargins(0, 0, 0, 0)
        swatch = QLabel()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #222; border-radius: 2px;")
        h_layout.addWidget(swatch)
        class_lbl = QLabel(class_name)
        class_lbl.setStyleSheet(f"font-weight: bold; color: {color_hex};")
        h_layout.addWidget(class_lbl)
        h_layout.addStretch()
        v.addWidget(h_class)

        conf_lbl = QLabel(f"Conf: **{confidence}**")
        v.addWidget(conf_lbl)

        coords_lbl = QLabel(f"Coords: {coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}")
        coords_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(coords_lbl)

        item_widget.setLayout(v)

        line = QLabel()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #DDDDDD;")

        self.bb_layout.addWidget(item_widget)
        self.bb_layout.addWidget(line)

        item_widget.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Maximum
        )

    def display_bounding_boxes(self, boxes):
        self.current_bboxes = boxes or []
        self._clear_bb_list()
        if not boxes:
            empty_lbl = QLabel("✅ No bounding boxes detected.")
            empty_lbl.setStyleSheet("color: green; padding: 6px; font-style: italic;")
            self.bb_layout.addWidget(empty_lbl)
            self.bb_layout.addStretch()
            return

        for box in boxes:
            cls = box.get("class", "unk")
            coords = box.get("coords", [0, 0, 0, 0])
            color = box.get("color", "#AAAAAA")
            conf = box.get("confidence", "N/A")
            self._add_bb_item(cls, coords, color, conf)

        self.bb_layout.addStretch()

    def _overlay_boxes_on_preview(self, boxes):
        if not self.current_pixmap:
            return None

        pixmap = QPixmap(self.current_image_path)

        if not boxes:
            scaled_original = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_original)
            return pixmap

        painted = QPixmap(pixmap)
        painter = QPainter(painted)
        painter.setRenderHint(QPainter.Antialiasing)

        for box in boxes:
            coords = box.get("coords", [0, 0, 0, 0])
            color_hex = box.get("color", "#FF0000")

            try:
                color = QColor(color_hex)
                painter.setPen(QColor(color.red(), color.green(), color.blue(), 255))
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 50))
                painter.setOpacity(1.0)
            except Exception:
                painter.setPen(QColor("#FF0000"))
                painter.setBrush(QColor(255, 0, 0, 50))

            x = int(coords[0])
            y = int(coords[1])
            w = int(coords[2] - coords[0])
            h = int(coords[3] - coords[1])

            painter.drawRect(x, y, max(1, w), max(1, h))

        painter.end()

        scaled_painted = painted.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_painted)

        return painted

    def run_segmentation(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Select an image first.")
            return

        model_name = self.seg_model_dropdown.currentText()
        self.category_label.setText(f"Running segmentation using **{model_name.upper()}**...")
        QApplication.processEvents()

        try:
            # 1) obtain mask from models.py. It returns a dictionary: 
            #    {'mask_img': PIL.Image, 'model_name_used': str}
            result = self.seg_models.segment(self.current_image_path, model_name)

            # --- FIX: Extract the PIL Image and the actual model name used ---
            mask_img = result["mask_img"]
            used_model_name = result["model_name_used"]
            
            # 2) create coloured result by multiplying original image by mask (soft multiplication)
            coloured = self.multiply_image_by_mask(self.current_image_path, mask_img, blur_radius=0.0)

            # 3) ensure output folder exists and save coloured image
            if not self.current_folder:
                self.current_folder = os.path.dirname(self.current_image_path)

            # Use the actual model name used (used_model_name) for accurate logging/filename
            coloured_name = f"{os.path.splitext(os.path.basename(self.current_image_path))[0]}_{used_model_name}_coloured_mask.png"
            self.segmentation_output_path = os.path.join(self.current_folder, coloured_name)
            coloured.save(self.segmentation_output_path)

            # 4) show coloured image in the Segmentation Output area
            # Scale based on output label size, which is now vertically constrained
            self.output_label.setPixmap(QPixmap(self.segmentation_output_path).scaled(
                self.output_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.output_label.setText("")
            self.output_label.setStyleSheet("border: 2px solid #00AA00; background-color: #F0FFF0;")
            
            # Use the actual model name used for accurate user feedback
            self.category_label.setText(f"Segmentation done using **{used_model_name.upper()}**. Coloured mask saved.")
            print(f"{used_model_name} coloured output saved to: {self.segmentation_output_path}")

        except Exception as e:
            # This is the error handling block. If the error is occurring here, 
            # it means the primary attempt failed and the next line is hit.
            # We don't change anything here, but we check if the error originated here.
            QMessageBox.critical(self, "Segmentation Error", str(e))
            self.output_label.setStyleSheet("border: 2px solid #AA0000; background-color: #FFF0F0;")
            self.output_label.setText("SEGMENTATION FAILED")
            self.category_label.setText(f"Segmentation Failed! Error: {e}")
            print("Error:", e)

    def handle_generate_report(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Select an image first.")
            return

        # ensure detection overlay exists
        if not self.detection_output_path:
            QMessageBox.warning(self, "Missing Detection", "Please run Detection/Categorize first to generate overlay for report.")
            return

        full_metadata = self.metadata_label.text()

        # Build a robust bounding_boxes list to pass to the report generator.
        # Priority:
        #  1) use self.current_bboxes if it's populated (preferred)
        #  2) fallback: parse the UI list (bb_layout) to reconstruct entries
        bounding_boxes_for_report = []

        # 1) prefer current_bboxes (should be structured dicts already)
        if getattr(self, "current_bboxes", None):
            # Defensive copy and normalize fields
            for bb in self.current_bboxes:
                if not isinstance(bb, dict):
                    continue
                cls = bb.get("class", "unknown")
                conf = bb.get("confidence", bb.get("conf", "N/A"))
                coords = bb.get("coords", bb.get("bbox", []))
                # try to coerce coords into list of 4 floats/ints
                try:
                    if isinstance(coords, (tuple, list)) and len(coords) == 4:
                        coords = [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]
                    else:
                        coords = []
                except Exception:
                    coords = []
                bounding_boxes_for_report.append({
                    "class": cls,
                    "confidence": conf,
                    "coords": coords
                })

        # 2) fallback: parse visible UI entries if current_bboxes was empty
        if not bounding_boxes_for_report:
            # each item_widget in bb_layout contains labels in a predictable order:
            # class label, conf label, coords label (we parse their text)
            for i in range(self.bb_layout.count()):
                widget = self.bb_layout.itemAt(i).widget()
                if widget is None:
                    continue
                # skip thin separator lines or container spacers
                # look for widgets that contain multiple QLabel children (our item_widget)
                labels = widget.findChildren(QLabel)
                if not labels:
                    continue

                # Heuristic parsing: find label texts that look like class/conf/coords
                class_name = None
                confidence = "N/A"
                coords = []
                for lbl in labels:
                    txt = lbl.text().strip()
                    # class label often contains the defect name (no colon), or a small swatch + name
                    if txt and not txt.lower().startswith("conf:") and not txt.lower().startswith("coords:"):
                        # first non-conf/coords label treat as class
                        if class_name is None:
                            # remove any HTML-ish markup
                            class_name = txt.splitlines()[0]
                            # remove leading icons if any
                            class_name = class_name.replace("\u25A0", "").strip()
                            continue
                    if txt.lower().startswith("conf:"):
                        # possible formats: "Conf: **0.86**" or "Conf: 0.86"
                        conf_part = txt.split(":", 1)[1].strip()
                        conf_part = conf_part.replace("*", "").strip()
                        confidence = conf_part
                    if txt.lower().startswith("coords:"):
                        coords_part = txt.split(":", 1)[1].strip()
                        # coords likely like "537, 709, 596, 755" or "[537, 709, 596, 755]"
                        coords_part = coords_part.strip("[] ")
                        try:
                            parts = [float(x.strip()) for x in coords_part.split(",") if x.strip() != ""]
                            if len(parts) == 4:
                                coords = parts
                        except Exception:
                            coords = []

                if class_name:
                    bounding_boxes_for_report.append({
                        "class": class_name,
                        "confidence": confidence,
                        "coords": coords
                    })

        # If still empty, warn and let report_generator handle no-defect case
        if not bounding_boxes_for_report:
            QMessageBox.information(self, "No Defects", "No detected defects were found to include in the report.")
        else:
            # Optional: normalize confidence to a float string with 2 decimals if numeric
            for bb in bounding_boxes_for_report:
                try:
                    bb_conf = float(str(bb.get("confidence")).replace("*", ""))
                    bb["confidence"] = f"{bb_conf:.2f}"
                except Exception:
                    bb["confidence"] = str(bb.get("confidence"))

        # Save an HTML copy (optional) so images referenced by PDF generator are in the same folder
        try:
            if not self.current_folder:
                self.current_folder = os.path.dirname(self.current_image_path)

            report_html_path = os.path.join(
                self.current_folder,
                f"{os.path.splitext(os.path.basename(self.current_image_path))[0]}_enhanced_report.html"
            )
            with open(report_html_path, "w", encoding="utf-8") as f:
                f.write("<html><head><meta charset='utf-8'><title>Enhanced Report</title></head><body>")
                f.write(f"<h2>Report for {os.path.basename(self.current_image_path)}</h2>")
                f.write("<h3>Metadata</h3>")
                f.write("<pre>" + full_metadata + "</pre>")
                f.write("<h3>Detection Overlay</h3>")
                if self.detection_output_path and os.path.exists(self.detection_output_path):
                    f.write(f"<img src='{os.path.basename(self.detection_output_path)}' style='max-width:100%;height:auto;'/>")
                f.write("<h3>Segmentation Output (Coloured Mask)</h3>")
                if self.segmentation_output_path and os.path.exists(self.segmentation_output_path):
                    f.write(f"<img src='{os.path.basename(self.segmentation_output_path)}' style='max-width:100%;height:auto;'/>")
                f.write("<h3>Detected Defects Table</h3>")
                f.write("<table border='1' cellpadding='6' cellspacing='0'>")
                f.write("<tr><th>Defect Type</th><th>Confidence</th><th>Coordinates [x1,y1,x2,y2]</th><th>Size (w x h)</th></tr>")
                if bounding_boxes_for_report:
                    for r in bounding_boxes_for_report:
                        coords = r.get("coords") or []
                        if isinstance(coords, (list, tuple)) and len(coords) == 4:
                            x1, y1, x2, y2 = coords
                            w = int(round(float(x2) - float(x1)))
                            h = int(round(float(y2) - float(y1)))
                            size_str = f"{w} x {h}"
                            coords_str = f"[{int(round(x1))}, {int(round(y1))}, {int(round(x2))}, {int(round(y2))}]"
                        else:
                            size_str = "N/A"
                            coords_str = "N/A"
                        f.write(f"<tr><td>{r.get('class')}</td><td>{r.get('confidence')}</td><td>{coords_str}</td><td>{size_str}</td></tr>")
                else:
                    f.write("<tr><td colspan='4'>No defects detected</td></tr>")
                f.write("</table>")
                f.write("<p>Generated by system.</p>")
                f.write("</body></html>")
        except Exception as e:
            print("Failed to write HTML report:", e)

        # Finally call the PDF generator with the constructed bounding_boxes list
        generate_report(
            self,
            self.current_image_path,
            full_metadata,
            detection_image_path=self.detection_output_path,
            segmentation_image_path=self.segmentation_output_path,
            bounding_boxes=bounding_boxes_for_report
        )


    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit',
                                     "Do you want to save your progress before exiting?", QMessageBox.Yes |
                                     QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            try:
                self.save_progress()
                event.accept()
            except Exception:
                event.ignore()
        elif reply == QMessageBox.No:
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    window = GUI()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
