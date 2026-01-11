import sys
import os
import datetime
import subprocess
import platform
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QMessageBox, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader

from utils import build_sparse_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_plots")
os.makedirs(SAVE_DIR, exist_ok=True)

DEFAULT_THRESHOLD = 10

class FTIRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FTIR Spectra Classifier")
        self.setFixedSize(960, 560)  # Fixed window size for absolute layout

        # Set background image (with absolute path for CSS)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-image: url({os.path.join(BASE_DIR, 'background.png').replace('\\', '/')});
                background-repeat: no-repeat;
                background-position: center;
                background-attachment: fixed;
                background-size: cover;
            }}
        """)

        # ---- Central Widget ----
        self.central = QWidget(self)
        self.setCentralWidget(self.central)

        # ---- Buttons (Absolute Positioning) ----
        self.load_button = QPushButton("Select Spectra", self.central)
        self.load_button.setGeometry(30, 30, 200, 44)
        self.load_button.clicked.connect(self.load_and_show_spectrum)

        self.run_button = QPushButton("Run Prediction", self.central)
        self.run_button.setGeometry(30, 90, 200, 44)
        self.run_button.clicked.connect(self.run_prediction)

        self.open_pred_button = QPushButton("Open Prediction Results", self.central)
        self.open_pred_button.setGeometry(30, 150, 200, 44)
        self.open_pred_button.clicked.connect(self.open_prediction_image)

        self.export_pdf_button = QPushButton("Export as PDF", self.central)
        self.export_pdf_button.setGeometry(30, 210, 200, 44)
        self.export_pdf_button.clicked.connect(self.export_as_pdf)

        self.open_last_saved_button = QPushButton("Open the Last Spectra", self.central)
        self.open_last_saved_button.setGeometry(30, 270, 200, 44)
        self.open_last_saved_button.clicked.connect(self.open_last_saved_image)

        # ---- Style Buttons  ----
        button_style = """
            QPushButton {
                background-color: #113F67;
                color: #fff;
                border-radius: 12px;
                border: none;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #113F67;
                color: #fff;
            }
        """
        self.load_button.setStyleSheet(button_style)
        self.run_button.setStyleSheet(button_style)
        self.open_pred_button.setStyleSheet(button_style)
        self.export_pdf_button.setStyleSheet(button_style)
        self.open_last_saved_button.setStyleSheet(button_style)

        # ---- Table Controls ----
        self.table_label = QLabel("📊 Probabilistic Table (Set threshold):", self.central)
        self.table_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.table_label.setGeometry(290, 25, 350, 32)
        self.table_label.setStyleSheet("color: #DCD0A8;")

        self.threshold_spinbox = QSpinBox(self.central)
        self.threshold_spinbox.setRange(1, 100)
        self.threshold_spinbox.setValue(DEFAULT_THRESHOLD)
        self.threshold_spinbox.setGeometry(650, 25, 60, 32)
        self.threshold_spinbox.valueChanged.connect(self.update_prediction_table)
        self.threshold_spinbox.setStyleSheet("""
            QSpinBox { color: #DCD0A8; background: #212121; border-radius: 8px; font-weight: 600; }
            QSpinBox::up-button { background: #333; }
            QSpinBox::down-button { background: #333; }
        """)

        threshold_label = QLabel("", self.central)
        threshold_label.setFont(QFont("Segoe UI", 10, QFont.Normal))
        threshold_label.setGeometry(575, 25, 80, 32)
        threshold_label.setStyleSheet("color: #DCD0A8;")

        # ---- Table Widget ----
        self.prediction_table = QTableWidget(0, 2, self.central)
        self.prediction_table.setGeometry(290, 60, 600, 350)
        self.prediction_table.setHorizontalHeaderLabels(["Compound Name", "Probability (%)"])
        self.prediction_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.prediction_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.prediction_table.setSelectionMode(QTableWidget.NoSelection)
        self.prediction_table.setFont(QFont("Segoe UI", 10))
        self.prediction_table.setShowGrid(False)
        self.prediction_table.setStyleSheet("""
            QTableWidget, QTableView {
                background: transparent;
                border: none;
            }
            QHeaderView::section {
                background-color: rgba(33,33,33,0.7); color: #DCD0A8; font-weight: bold;
                border: none;
                height: 30px;
            }
        """)

        # ---- Status Label ----
        self.status_label = QLabel("Status: Ready", self.central)
        self.status_label.setFont(QFont("Segoe UI", 11, QFont.Normal))
        self.status_label.setStyleSheet("font-style: italic; color: #333;")
        self.status_label.setGeometry(290, 430, 600, 32)
        self.status_label.setAlignment(Qt.AlignLeft)

        # Internal Data
        self.csv_path = None
        self.prediction_image_path = None
        self.spectrum_image_path = None
        self.probs = None
        self.classes = None

    def load_and_show_spectrum(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        self.csv_path = path
        self.status_label.setText(f"Status: Loaded {os.path.basename(path)}")

        try:
            raw = pd.read_csv(path, header=None, delimiter=';', dtype=str)
            wav = raw.iloc[2:, 0].str.replace(',', '.').astype(float).values
            abs_raw = raw.iloc[2:, 1].str.replace(',', '.').astype(float).values

            now = datetime.datetime.now()
            base = os.path.splitext(os.path.basename(path))[0]
            filename = f"{base}_{now.hour}_{now.minute}_{now.day}_{now.month}_spectrum.png"
            img_path = os.path.abspath(os.path.join(SAVE_DIR, filename))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wav, y=abs_raw, mode='lines'))
            fig.update_layout(
                title="Original Spectrum",
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Absorbance",
                xaxis_autorange='reversed',
                font=dict(size=22),
                xaxis=dict(title_font=dict(size=20), tickfont=dict(size=18)),
                yaxis=dict(title_font=dict(size=20), tickfont=dict(size=18))
            )
            fig.write_image(img_path, width=2400, height=1600, scale=2)
            fig.show()

            self.spectrum_image_path = img_path
            self.status_label.setText(f"Status: Spectrum saved to {os.path.basename(img_path)}")
            self.open_with_viewer(img_path)

        except Exception as e:
            self.status_label.setText("Status: Spectrum load failed.")
            QMessageBox.critical(self, "Error", str(e))

    def run_prediction(self):
        if not self.csv_path:
            QMessageBox.warning(self, "No file", "Please select a CSV file first.")
            return
        try:
            self.status_label.setText("Status: Running prediction...")

            # Load the sparse PLS pipeline (the one created by train_sparse_all_models.py)
            pipe = joblib.load("ftir_model_pipeline.pkl")
            pca = pipe["pca"]
            pls = pipe["pls"]
            lb = pipe["lb"]
            cols = pipe["columns"]

            # Read spectrum from CSV
            df = pd.read_csv(self.csv_path, header=None, delimiter=';')

            # Wavenumber axis and absorbance from THIS file
            wav_csv = (
                df.iloc[2:, 0]
                .astype(str)
                .str.replace(',', '.')
                .astype(float)
                .values
            )
            abs_raw = (
                df.iloc[2:, 1]
                .astype(str)
                .str.replace(',', '.')
                .astype(float)
                .values
            )

            # Safety check (just in case)
            if len(wav_csv) != len(abs_raw):
                raise ValueError(
                    f"Length mismatch between wavenumber and absorbance in CSV: "
                    f"{len(wav_csv)} vs {len(abs_raw)}"
                )

            # --- use the SAME sparse feature extraction as in training ---
            feats = build_sparse_features(abs_raw, wav_csv)

            # Align to training feature space
            X_unk = (
                pd.DataFrame([feats])
                .reindex(columns=cols, fill_value=0)
                .values
            )

            # PCA + PLS-DA
            Xp = pca.transform(X_unk)
            self.probs = pls.predict(Xp).ravel()
            self.classes = lb.classes_

            # Update table in the GUI
            self.update_prediction_table()

            # --- save bar-plot with scores ---
            now = datetime.datetime.now()
            base = os.path.splitext(os.path.basename(self.csv_path))[0]
            filename = f"{base}_{now.hour}_{now.minute}_{now.day}_{now.month}_prediction.png"
            path = os.path.abspath(os.path.join(SAVE_DIR, filename))

            fig = go.Figure()
            fig.add_trace(go.Bar(x=self.classes, y=self.probs, name="PLS-DA Score"))
            fig.update_layout(
                title="Prediction Scores",
                yaxis_title="Score",
                font=dict(size=22),
                xaxis=dict(title_font=dict(size=20), tickfont=dict(size=18)),
                yaxis=dict(title_font=dict(size=20), tickfont=dict(size=18))
            )
            fig.write_image(path, width=2400, height=1600, scale=2)
            fig.show()

            self.prediction_image_path = path
            self.status_label.setText(f"Status: Saved to {os.path.basename(path)}")
            self.open_with_viewer(path)

        except Exception as e:
            self.status_label.setText("Status: Prediction failed.")
            QMessageBox.critical(self, "Error", str(e))

    def update_prediction_table(self):
        self.prediction_table.setRowCount(0)
        threshold = self.threshold_spinbox.value() / 100

        classes = self.classes if self.classes is not None else []
        probs = self.probs if self.probs is not None else []

        top = sorted([(c, p) for c, p in zip(classes, probs) if p >= threshold],
                     key=lambda x: x[1], reverse=True)[:10]

        if not top:
            return

        for i, (cls, prob) in enumerate(top):
            self.prediction_table.insertRow(i)
            name_item = QTableWidgetItem(cls)
            prob_item = QTableWidgetItem(f"{prob * 100:.2f}")
            # Highlight only result cells
            name_item.setBackground(QColor("#D3ECCD"))
            prob_item.setBackground(QColor("#D3ECCD"))
            name_item.setForeground(QColor("#364fc7"))  # For contrast
            prob_item.setForeground(QColor("#364fc7"))
            self.prediction_table.setItem(i, 0, name_item)
            self.prediction_table.setItem(i, 1, prob_item)

    def export_as_pdf(self):
        if not self.spectrum_image_path or not self.prediction_image_path:
            QMessageBox.warning(self, "Missing Images", "You must generate both spectrum and prediction plots first.")
            return
        try:
            now = datetime.datetime.now()
            filename = f"report_{now.hour}_{now.minute}_{now.day}_{now.month}.pdf"
            pdf_path = os.path.join(SAVE_DIR, filename)

            c = canvas.Canvas(pdf_path, pagesize=landscape(A4))
            width, height = landscape(A4)
            margin = 50
            image_width = width - 2 * margin
            image_height = 200

            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin, height - 50, "FTIR Spectra Classification Report")
            c.setFont("Helvetica", 12)
            c.drawString(margin, height - 80, f"Generated on: {now.strftime('%Y-%m-%d %H:%M:%S')}")

            y = height - 120

            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, "Original Spectrum:")
            y -= 20
            c.drawImage(ImageReader(self.spectrum_image_path), margin, y - image_height,
                        width=image_width, height=image_height, preserveAspectRatio=True, mask='auto')
            y -= (image_height + 40)

            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, "Prediction Result:")
            y -= 20
            c.drawImage(ImageReader(self.prediction_image_path), margin, y - image_height,
                        width=image_width, height=image_height, preserveAspectRatio=True, mask='auto')

            c.showPage()
            c.save()
            self.status_label.setText(f"Status: PDF saved to {os.path.basename(pdf_path)}")
            self.open_with_viewer(pdf_path)

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def open_prediction_image(self):
        if self.prediction_image_path and os.path.exists(self.prediction_image_path):
            self.open_with_viewer(self.prediction_image_path)
        else:
            QMessageBox.warning(self, "Not Found", "No prediction image found.")

    def open_last_saved_image(self):
        try:
            files = [f for f in os.listdir(SAVE_DIR) if f.lower().endswith(".png")]
            if not files:
                QMessageBox.warning(self, "No Files", "No saved spectra found in the folder.")
                return
            files.sort(key=lambda f: os.path.getmtime(os.path.join(SAVE_DIR, f)), reverse=True)
            latest_file = os.path.join(SAVE_DIR, files[0])
            self.open_with_viewer(latest_file)
            self.status_label.setText(f"Status: Opened {os.path.basename(latest_file)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def open_with_viewer(self, path):
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.call(["open", path])
            else:
                subprocess.call(["xdg-open", path])
        except Exception as e:
            QMessageBox.critical(self, "Viewer Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FTIRApp()
    window.show()
    sys.exit(app.exec_())
