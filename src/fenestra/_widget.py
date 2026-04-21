import os
import glob
import tempfile
import pathlib

from qtpy.QtWidgets import QVBoxLayout, QWidget, QFileDialog, QMessageBox, QGroupBox, QFormLayout, QPushButton, QLabel, QComboBox, QLineEdit, QDoubleSpinBox, QCheckBox
from qtpy.QtCore import Qt
from magicgui import magic_factory
from magicgui.widgets import FileEdit, Container, PushButton, ComboBox, LineEdit, Label, FloatSpinBox
import tifffile
import numpy as np
import napari

from .pipeline import process_jpk, upsample_clahe, run_dl_upsampling, run_cellpose, quantify_fenestrations, run_batch_pipeline

class FenestraWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # State
        self.raw_image = None
        self.pixel_to_nm = None
        self.upsampled_image = None
        self.mask_image = None
        self.temp_dir = tempfile.TemporaryDirectory()
        
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # ---------------------------------------------
        # 1. IO Group
        # ---------------------------------------------
        io_group = QGroupBox("1. Input Data")
        io_layout = QFormLayout()
        
        self.btn_load_jpk = QPushButton("Load JPK.qi-image")
        self.btn_load_jpk.clicked.connect(self.on_load_jpk)
        io_layout.addRow("Raw File:", self.btn_load_jpk)
        
        self.lbl_info = QLabel("No file loaded.")
        io_layout.addRow("", self.lbl_info)
        
        io_group.setLayout(io_layout)
        main_layout.addWidget(io_group)
        
        # ---------------------------------------------
        # 2. Upsampling Group
        # ---------------------------------------------
        up_group = QGroupBox("2. Upsampling / Enhancement")
        up_layout = QFormLayout()
        
        self.combo_method = QComboBox()
        self.combo_method.addItems(["CLAHE (CPU)", "HAT", "SwinIR"])
        up_layout.addRow("Method:", self.combo_method)
        
        # Factor (only for CLAHE)
        self.factor_widget = QWidget()
        factor_layout = QFormLayout()
        factor_layout.setContentsMargins(0, 0, 0, 0)
        self.factor_widget.setLayout(factor_layout)
        
        self.spin_up_factor = QDoubleSpinBox()
        self.spin_up_factor.setDecimals(0)
        self.spin_up_factor.setRange(1, 10)
        self.spin_up_factor.setValue(4)
        factor_layout.addRow("Factor:", self.spin_up_factor)
        up_layout.addRow(self.factor_widget)
        
        # Postprocess params group
        self.postprocess_widget = QWidget()
        pp_layout = QFormLayout()
        pp_layout.setContentsMargins(0, 0, 0, 0)
        self.postprocess_widget.setLayout(pp_layout)
        
        self.spin_clahe_clip = QDoubleSpinBox()
        self.spin_clahe_clip.setDecimals(3)
        self.spin_clahe_clip.setSingleStep(0.01)
        self.spin_clahe_clip.setValue(0.02) # Moderate default
        pp_layout.addRow("Clip Limit:", self.spin_clahe_clip)
        
        self.spin_unsharp_radius = QDoubleSpinBox()
        self.spin_unsharp_radius.setValue(1.0)
        pp_layout.addRow("Unsharp Radius:", self.spin_unsharp_radius)
        
        self.spin_unsharp_amount = QDoubleSpinBox()
        self.spin_unsharp_amount.setValue(1.0)
        pp_layout.addRow("Unsharp Amount:", self.spin_unsharp_amount)
        
        up_layout.addRow(self.postprocess_widget)
        
        # DL params group
        self.dl_widget = QWidget()
        dl_layout = QFormLayout()
        dl_layout.setContentsMargins(0, 0, 0, 0)
        self.dl_widget.setLayout(dl_layout)

        # Model path
        self.line_model_path = QLineEdit("/home/arka/Desktop/AFM-Project/DL_Upsampling/models/best_model_ema.pth")
        self.btn_pick_model = QPushButton("...")
        self.btn_pick_model.setFixedWidth(30)
        self.btn_pick_model.clicked.connect(lambda: self.pick_file(self.line_model_path, "*.pth"))
        
        path_layout = QVBoxLayout()
        path_layout.addWidget(self.line_model_path)
        path_layout.addWidget(self.btn_pick_model)
        dl_layout.addRow("DL Model:", path_layout)
        
        # Engine
        self.combo_engine = QComboBox()
        self.combo_engine.addItems(["Singularity", "Docker"])
        self.combo_engine.currentIndexChanged.connect(self.on_engine_changed)
        dl_layout.addRow("Engine:", self.combo_engine)

        # DL Post-process checkbox
        self.chk_postprocess = QCheckBox("Apply Post-DL Sharpening")
        self.chk_postprocess.toggled.connect(self.on_method_changed)
        dl_layout.addRow("", self.chk_postprocess)

        # Container path / tag
        self.lbl_container = QLabel("Singularity (.sif):")
        self.line_container = QLineEdit("/home/arka/Desktop/AFM-Project/DL_Upsampling/containers/dl_upsampling.sif")
        self.btn_pick_container = QPushButton("...")
        self.btn_pick_container.setFixedWidth(30)
        self.btn_pick_container.clicked.connect(lambda: self.pick_file(self.line_container, "*.sif *.def"))
        
        cont_layout = QVBoxLayout()
        cont_layout.addWidget(self.line_container)
        cont_layout.addWidget(self.btn_pick_container)
        dl_layout.addRow(self.lbl_container, cont_layout)
        
        up_layout.addRow(self.dl_widget)
        
        self.combo_method.currentIndexChanged.connect(self.on_method_changed)
        self.on_method_changed() # set initial visibility
        
        self.btn_run_up = QPushButton("Run Upsampling")
        self.btn_run_up.clicked.connect(self.on_run_upsampling)
        up_layout.addRow("", self.btn_run_up)
        
        up_group.setLayout(up_layout)
        main_layout.addWidget(up_group)
        
        # ---------------------------------------------
        # 3. Segmentation Group
        # ---------------------------------------------
        seg_group = QGroupBox("3. Cellpose Segmentation")
        seg_layout = QFormLayout()
        
        self.line_cp_model = QLineEdit("") # empty means cyto2
        self.line_cp_model.setPlaceholderText("Leave empty for cyto2")
        self.btn_pick_cp = QPushButton("...")
        self.btn_pick_cp.setFixedWidth(30)
        self.btn_pick_cp.clicked.connect(lambda: self.pick_file(self.line_cp_model, "*"))
        
        cp_layout = QVBoxLayout()
        cp_layout.addWidget(self.line_cp_model)
        cp_layout.addWidget(self.btn_pick_cp)
        seg_layout.addRow("CP Model:", cp_layout)
        
        self.spin_diameter = QDoubleSpinBox()
        self.spin_diameter.setRange(0.0, 500.0)
        self.spin_diameter.setValue(30.0)
        seg_layout.addRow("Diameter (0=auto):", self.spin_diameter)
        
        self.spin_cellprob = QDoubleSpinBox()
        self.spin_cellprob.setRange(-10.0, 10.0)
        self.spin_cellprob.setSingleStep(0.1)
        self.spin_cellprob.setValue(0.0)
        seg_layout.addRow("Cellprob Thresh:", self.spin_cellprob)
        
        self.spin_flow = QDoubleSpinBox()
        self.spin_flow.setRange(0.0, 10.0)
        self.spin_flow.setSingleStep(0.1)
        self.spin_flow.setValue(0.4)
        seg_layout.addRow("Flow Thresh:", self.spin_flow)
        
        self.btn_run_cp = QPushButton("Run Cellpose")
        self.btn_run_cp.clicked.connect(self.on_run_cellpose)
        seg_layout.addRow("", self.btn_run_cp)
        
        seg_group.setLayout(seg_layout)
        main_layout.addWidget(seg_group)
        
        # ---------------------------------------------
        # 4. Analysis and Views
        # ---------------------------------------------
        an_group = QGroupBox("4. Layout & Analysis")
        an_layout = QVBoxLayout()
        
        self.btn_sync_layout = QPushButton("Arrange 4-Pane Grid")
        self.btn_sync_layout.clicked.connect(self.on_sync_layout)
        an_layout.addWidget(self.btn_sync_layout)
        
        self.btn_quantify = QPushButton("Quantify Fenestrations")
        self.btn_quantify.clicked.connect(self.on_quantify)
        an_layout.addWidget(self.btn_quantify)
        
        an_group.setLayout(an_layout)
        main_layout.addWidget(an_group)
        
        # ---------------------------------------------
        # 5. Batch Analysis
        # ---------------------------------------------
        batch_group = QGroupBox("5. Batch Analysis")
        batch_layout = QFormLayout()
        
        # Input directory
        self.line_batch_input = QLineEdit()
        self.line_batch_input.setPlaceholderText("Folder containing .jpk-qi-image files")
        self.btn_batch_input = QPushButton("Browse")
        self.btn_batch_input.clicked.connect(lambda: self._pick_dir(self.line_batch_input))
        batch_input_layout = QVBoxLayout()
        batch_input_layout.addWidget(self.line_batch_input)
        batch_input_layout.addWidget(self.btn_batch_input)
        batch_layout.addRow("Input Dir:", batch_input_layout)
        
        # Output directory
        self.line_batch_output = QLineEdit()
        self.line_batch_output.setPlaceholderText("Folder for results (Excel + TIFFs)")
        self.btn_batch_output = QPushButton("Browse")
        self.btn_batch_output.clicked.connect(lambda: self._pick_dir(self.line_batch_output))
        batch_output_layout = QVBoxLayout()
        batch_output_layout.addWidget(self.line_batch_output)
        batch_output_layout.addWidget(self.btn_batch_output)
        batch_layout.addRow("Output Dir:", batch_output_layout)
        
        # Progress label
        self.lbl_batch_progress = QLabel("Idle")
        batch_layout.addRow("Status:", self.lbl_batch_progress)
        
        # Run button
        self.btn_run_batch = QPushButton("Run Batch")
        self.btn_run_batch.clicked.connect(self.on_run_batch)
        batch_layout.addRow("", self.btn_run_batch)
        
        batch_group.setLayout(batch_layout)
        main_layout.addWidget(batch_group)
        
        main_layout.addStretch()
        
    def pick_file(self, line_edit, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_str)
        if path:
            line_edit.setText(path)

    def on_method_changed(self):
        method = self.combo_method.currentText()
        is_clahe = "CLAHE" in method
        
        self.factor_widget.setVisible(is_clahe)
        self.dl_widget.setVisible(not is_clahe)
        
        if is_clahe:
            self.postprocess_widget.setVisible(True)
        else:
            self.postprocess_widget.setVisible(self.chk_postprocess.isChecked())

    def on_engine_changed(self):
        engine = self.combo_engine.currentText()
        if engine == "Docker":
            self.lbl_container.setText("Docker Tag:")
            self.line_container.setText("livrvub/dl-upsampling:latest")
            self.btn_pick_container.setVisible(False)
        else:
            self.lbl_container.setText("Singularity (.sif):")
            self.line_container.setText("/home/arka/Desktop/AFM-Project/DL_Upsampling/containers/dl_upsampling.sif")
            self.btn_pick_container.setVisible(True)

    # --- Actions ---

    def on_load_jpk(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select JPK", "", "JPK Files (*.jpk *.jpk-qi-image)")
        if not path:
            return
            
        try:
            self.raw_image, self.pixel_to_nm = process_jpk(path)
            self.lbl_info.setText(f"Size: {self.raw_image.shape}\nScale: {self.pixel_to_nm:.2f} nm/px")
            
            # Remove old
            if "Raw AFM" in self.viewer.layers:
                self.viewer.layers.remove("Raw AFM")
                
            self.viewer.add_image(
                self.raw_image,
                name="Raw AFM",
                colormap="magma"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load JPK:\n{e}")

    def on_run_upsampling(self):
        if self.raw_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
            
        method = self.combo_method.currentText()
        self.btn_run_up.setText("Upsampling in progress...")
        self.btn_run_up.setEnabled(False)
        
        if "CLAHE" in method:
            # Run local fallback
            try:
                factor = int(self.spin_up_factor.value())
                clip = self.spin_clahe_clip.value()
                uradius = self.spin_unsharp_radius.value()
                uamount = self.spin_unsharp_amount.value()
                self.upsampled_image = upsample_clahe(self.raw_image, factor, clip, uradius, uamount)
                self.finalize_upsampling()
            except Exception as e:
                self.btn_run_up.setText("Run Upsampling")
                self.btn_run_up.setEnabled(True)
                QMessageBox.critical(self, "Error", str(e))
        else:
            # Run Deep Learning
            model_path = self.line_model_path.text()
            container_path = self.line_container.text()
            arch = "hat" if "HAT" in method else "swinir"
            
            engine = self.combo_engine.currentText()
            if not os.path.exists(model_path):
                QMessageBox.critical(self, "Error", "Model path is invalid.")
                self.btn_run_up.setText("Run Upsampling")
                self.btn_run_up.setEnabled(True)
                return
                
            if engine == "Singularity" and not os.path.exists(container_path):
                QMessageBox.critical(self, "Error", "Singularity container path is invalid.")
                self.btn_run_up.setText("Run Upsampling")
                self.btn_run_up.setEnabled(True)
                return
                
            # Write temp input
            temp_in_path = os.path.join(self.temp_dir.name, "temp_in.tif")
            tifffile.imwrite(temp_in_path, self.raw_image)
            temp_out_dir = os.path.join(self.temp_dir.name, "out")
            os.makedirs(temp_out_dir, exist_ok=True)
            
            # Start worker
            # Note: the worker function is a generator yielding the final string.
            worker = run_dl_upsampling(
                temp_in_path=temp_in_path,
                temp_out_dir=temp_out_dir,
                container_path=container_path,
                model_path=model_path,
                architecture=arch,
                engine=self.combo_engine.currentText()
            )
            
            worker.yielded.connect(self._on_dl_complete)
            worker.errored.connect(self._on_dl_error)
            worker.start()
            
    def _on_dl_complete(self, result):
        import glob
        # Find the output tiff in temp_out_dir
        temp_out_dir = os.path.join(self.temp_dir.name, "out")
        out_files = glob.glob(os.path.join(temp_out_dir, "*.tif*"))
        if not out_files:
            self._on_dl_error(RuntimeError("No output generated from DL Upsampling."))
            return
            
        self.upsampled_image = tifffile.imread(out_files[0])
        
        if self.chk_postprocess.isChecked():
            clip = self.spin_clahe_clip.value()
            uradius = self.spin_unsharp_radius.value()
            uamount = self.spin_unsharp_amount.value()
            from fenestra.pipeline import apply_post_processing
            self.upsampled_image = apply_post_processing(
                self.upsampled_image, 
                clip_limit=clip, 
                unsharp_radius=uradius, 
                unsharp_amount=uamount
            )
            
        self.finalize_upsampling()

    def _on_dl_error(self, err):
        self.btn_run_up.setText("Run Upsampling")
        self.btn_run_up.setEnabled(True)
        QMessageBox.critical(self, "DL Error", str(err))

    def finalize_upsampling(self):
        self.btn_run_up.setText("Run Upsampling")
        self.btn_run_up.setEnabled(True)
        
        if "Upsampled AFM" in self.viewer.layers:
            self.viewer.layers.remove("Upsampled AFM")
            
        self.viewer.add_image(
            self.upsampled_image,
            name="Upsampled AFM",
            colormap="magma",
            scale=(0.25, 0.25) # assumed 4x upsampling
        )

    def on_run_cellpose(self):
        if self.upsampled_image is None:
            QMessageBox.warning(self, "Warning", "Please run upsampling first (or load one natively).")
            return
            
        self.btn_run_cp.setText("Segmenting...")
        self.btn_run_cp.setEnabled(False)
        
        model_p = self.line_cp_model.text()
        diam = self.spin_diameter.value()
        cprob = self.spin_cellprob.value()
        flow = self.spin_flow.value()
        
        worker = run_cellpose(self.upsampled_image, model_p, diam, cprob, flow)
        worker.yielded.connect(self._on_cp_complete)
        worker.errored.connect(self._on_cp_error)
        worker.start()
        
    def _on_cp_complete(self, masks):
        self.btn_run_cp.setText("Run Cellpose")
        self.btn_run_cp.setEnabled(True)
        self.mask_image = masks
        
        if "Cellpose Masks" in self.viewer.layers:
            self.viewer.layers.remove("Cellpose Masks")
            
        self.viewer.add_labels(
            self.mask_image,
            name="Cellpose Masks",
            scale=(0.25, 0.25)
        )
        
    def _on_cp_error(self, err):
        self.btn_run_cp.setText("Run Cellpose")
        self.btn_run_cp.setEnabled(True)
        QMessageBox.critical(self, "Cellpose Error", str(err))

    def on_sync_layout(self):
        # We want to show: Raw, Upsampled, Mask, Overlay (Upsampled + Mask boundary)
        
        # Calculate overlay if we have upsampled and mask
        if self.upsampled_image is not None and self.mask_image is not None:
            if "Overlay" in self.viewer.layers:
                self.viewer.layers.remove("Overlay")
            
            from skimage.segmentation import find_boundaries
            boundaries = find_boundaries(self.mask_image, mode='inner')
            
            # We just add boundaries as labels over the Upsampled AFM directly in the 4th view.
            # But Napari grid view works on *all* layers. 
            # To get 4 distinct views, we need 4 distinct layers:
            # Layer 1: Raw AFM (scale 1x)
            # Layer 2: Upsampled (scale 0.25x)
            # Layer 3: Mask (scale 0.25x)
            # Layer 4: We can create an RGB overlay image.
            
            # Simple RGB overlay: Red for boundaries
            base = self.upsampled_image.astype(np.float32)
            base = (base - base.min()) / (base.max() - base.min() + 1e-8)
            base = (base * 255).astype(np.uint8)
            
            rgb = np.stack([base, base, base], axis=-1)
            rgb[boundaries] = [255, 0, 0] # Red boundaries
            
            self.viewer.add_image(rgb, name="Overlay", scale=(0.25, 0.25), rgb=True)
            
        self.viewer.grid.enabled = True
        self.viewer.grid.shape = (2, 2)
        self.viewer.reset_view()

    def _pick_dir(self, line_edit):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            line_edit.setText(directory)

    def on_quantify(self):
        if self.mask_image is None or self.pixel_to_nm is None:
            QMessageBox.warning(self, "Warning", "Requires JPK loaded and Segmented Masks.")
            return
            
        try:
            method = self.combo_method.currentText()
            factor = self.spin_up_factor.value() if "CLAHE" in method else 4.0
            
            upsampled_scale = self.pixel_to_nm / factor
            df, porosity = quantify_fenestrations(self.mask_image, upsampled_scale, factor)
            
            # Ask where to save
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Metrics", "fenestration_metrics.csv", "CSV Files (*.csv)")
            if save_path:
                df.to_csv(save_path, index=False)
                QMessageBox.information(self, "Success", f"Saved {len(df)} fenestrations.\nOverall Porosity: {porosity*100:.2f}%")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to quantify: {e}")

    # --- Batch Analysis ---

    def on_run_batch(self):
        input_dir = self.line_batch_input.text().strip()
        output_dir = self.line_batch_output.text().strip()
        
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Warning", "Please select a valid input directory.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Warning", "Please select an output directory.")
            return
        
        method = self.combo_method.currentText()
        is_dl = "CLAHE" not in method
        
        # Validate DL paths if needed
        if is_dl:
            model_path = self.line_model_path.text().strip()
            if not os.path.exists(model_path):
                QMessageBox.critical(self, "Error", "DL Model path is invalid.")
                return
            engine = self.combo_engine.currentText()
            container_path = self.line_container.text().strip()
            if engine == "Singularity" and not os.path.exists(container_path):
                QMessageBox.critical(self, "Error", "Singularity container path is invalid.")
                return
        
        self.btn_run_batch.setText("Batch in progress...")
        self.btn_run_batch.setEnabled(False)
        self.lbl_batch_progress.setText("Starting...")
        
        worker = run_batch_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            method=method,
            clahe_factor=int(self.spin_up_factor.value()),
            clip_limit=self.spin_clahe_clip.value(),
            unsharp_radius=self.spin_unsharp_radius.value(),
            unsharp_amount=self.spin_unsharp_amount.value(),
            dl_model_path=self.line_model_path.text().strip() if is_dl else "",
            container_path=self.line_container.text().strip() if is_dl else "",
            engine=self.combo_engine.currentText() if is_dl else "Singularity",
            apply_postprocess=self.chk_postprocess.isChecked() if is_dl else False,
            cp_model_path=self.line_cp_model.text().strip(),
            diameter=self.spin_diameter.value(),
            cellprob_threshold=self.spin_cellprob.value(),
            flow_threshold=self.spin_flow.value(),
        )
        
        worker.yielded.connect(self._on_batch_progress)
        worker.errored.connect(self._on_batch_error)
        worker.start()

    def _on_batch_progress(self, msg):
        if isinstance(msg, str) and msg.startswith("BATCH_COMPLETE:"):
            total = msg.split(":")[1]
            self.btn_run_batch.setText("Run Batch")
            self.btn_run_batch.setEnabled(True)
            self.lbl_batch_progress.setText(f"Complete — {total} images processed.")
            QMessageBox.information(
                self, "Batch Complete",
                f"Successfully processed {total} images.\n\n"
                f"Results saved to:\n{self.line_batch_output.text()}"
            )
        else:
            self.lbl_batch_progress.setText(str(msg))

    def _on_batch_error(self, err):
        self.btn_run_batch.setText("Run Batch")
        self.btn_run_batch.setEnabled(True)
        self.lbl_batch_progress.setText("Error — see details.")
        QMessageBox.critical(self, "Batch Error", str(err))

