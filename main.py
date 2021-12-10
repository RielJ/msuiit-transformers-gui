import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PyQt6 import QtCore, QtGui, QtWidgets
from torch.backends import cudnn

import ui_main
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.datasets import IMG_FORMATS, VID_FORMATS
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory


class TransformerApp(ui_main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(TransformerApp, self).__init__()
        self.timer_video = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.setupUi(self)
        self.showMaximized()
        self.setWindowTitle("Distribution Line Detection")
        self.input_import.setVisible(not self.input_import.isVisible())
        self.label_import.setVisible(not self.label_import.isVisible())
        self.init_yolov5_defaults()
        self.init_slots()
        self.class_checkboxes = [
            self.class_0,
            self.class_1,
            self.class_2,
            self.class_3,
            self.class_4,
            self.class_5,
        ]
        background_image = cv2.imread("assets/background.png")
        resized_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2BGRA)
        resized_image = cv2.resize(
            resized_image, (640, 480), interpolation=cv2.INTER_AREA
        )
        self.background_image = QtGui.QImage(
            resized_image.data,
            resized_image.shape[1],
            resized_image.shape[0],
            QtGui.QImage.Format.Format_RGB32,
        )
        self.input_image.setPixmap(QtGui.QPixmap.fromImage(self.background_image))
        self.preview_group = QtWidgets.QButtonGroup()
        self.preview_group.addButton(self.before_button)
        self.preview_group.addButton(self.after_button)

    def init_slots(self):
        self.weight_directories = [
            x for x in Path(ROOT).glob("**/weights/*") if x.is_dir()
        ]
        model_types = [
            " ".join(word.title() for word in x.name.split("_"))
            for x in self.weight_directories
        ]
        self.input_detection_type.addItems(model_types)
        self.input_detection_type.setCurrentIndex(0)
        self.detection_type_changed()
        self.input_detection_type.textActivated.connect(self.detection_type_changed)
        self.input_mode.textActivated.connect(self.mode_input_changed)
        self._open_folder_detection_action = self.input_import.addAction(
            QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon
            ),
            QtWidgets.QLineEdit.ActionPosition.TrailingPosition,
        )
        self._open_folder_detection_action.triggered.connect(self.import_file_open)
        self.input_import.textChanged.connect(self.on_import_change)
        self.before_button.clicked.connect(self.on_before)
        self.after_button.clicked.connect(self.on_after)
        self.pause_button.clicked.connect(self.on_pause)
        self.resume_button.clicked.connect(self.on_resume)
        self.stop_button.clicked.connect(self.on_stop)
        self.cancel_button.clicked.connect(self.on_cancel)
        self.save_button.clicked.connect(self.on_save)
        self.detect_button.clicked.connect(self.on_detect)
        self.timer_video.timeout.connect(self.show_video_frame)
        self.hide_all_buttons()
        self.detect_button.setHidden(False)
        self.actionExit.setShortcut("Ctrl+Q")
        self.actionExit.setStatusTip("Exit application")
        self.actionExit.triggered.connect(QtWidgets.QApplication.quit)
        self.class_0.stateChanged.connect(lambda: self.checkbox_onchange())
        self.class_1.stateChanged.connect(lambda: self.checkbox_onchange())
        self.class_2.stateChanged.connect(lambda: self.checkbox_onchange())
        self.class_3.stateChanged.connect(lambda: self.checkbox_onchange())
        self.class_4.stateChanged.connect(lambda: self.checkbox_onchange())
        self.class_5.stateChanged.connect(lambda: self.checkbox_onchange())

    def hide_all_buttons(self):
        self.cancel_button.setHidden(True)
        self.before_button.setHidden(True)
        self.after_button.setHidden(True)
        self.pause_button.setHidden(True)
        self.resume_button.setHidden(True)
        self.save_button.setHidden(True)
        self.stop_button.setHidden(True)
        self.detect_button.setHidden(True)

    def reset_occurences(self):
        self.occ_class_0.setText("0")
        self.occ_class_1.setText("0")
        self.occ_class_2.setText("0")
        self.occ_class_3.setText("0")
        self.occ_class_4.setText("0")
        self.occ_class_5.setText("0")

    def on_save(self):
        if self.input_mode.currentText() == "Image":
            saved_name = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Image Output",
                "",
                "Image (*.jpg *.jpeg *.png);;All Files(*)",
            )
            os.rename(f"{ROOT}/prediction.jpg", saved_name[0])
        else:
            saved_name = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Video Output", "", "Video (*.mp4 *avi);;All Files(*)"
            )
            os.rename(f"{ROOT}/result.avi", saved_name[0])
        self.hide_all_buttons()
        self.detect_button.setHidden(False)
        self.input_image.setPixmap(QtGui.QPixmap.fromImage(self.background_image))
        self.reset_occurences()
        self.statusBar.showMessage("Output Saved!")
        self.statusBar.repaint()

    def on_before(self):
        _preview_image = cv2.imread(self.img_name)
        resized_image = self.resize_img(_preview_image)
        self.input_image.setPixmap(resized_image)

    def on_after(self):
        self.input_image.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def on_cancel(self):
        self.input_image.clear()
        self.input_image.setPixmap(QtGui.QPixmap.fromImage(self.background_image))
        self.hide_all_buttons()
        self.reset_occurences()
        self.detect_button.setHidden(False)
        self.statusBar.showMessage("Canceled!")
        self.statusBar.repaint()

    def on_import_change(self):
        if not self.input_import.text():
            self.input_image.clear()
            self.input_image.setPixmap(QtGui.QPixmap.fromImage(self.background_image))

    def detection_type_changed(self):
        self.input_detection_model.clear()
        self.file_models = [
            x
            for x in self.weight_directories[
                self.input_detection_type.currentIndex()
            ].glob("**/*.pt")
            if x.is_file()
        ]
        models = [os.path.splitext(x.name)[0] for x in self.file_models]
        self.input_detection_model.addItems(models)

    def checkbox_onchange(self):
        self.classes = [
            idx for idx, val in enumerate(self.class_checkboxes) if val.isChecked()
        ]
        unchecked = [
            idx for idx, val in enumerate(self.class_checkboxes) if not val.isChecked()
        ]
        for index in unchecked:
            getattr(
                self,
                f"occ_class_{index}",
            ).setText("0")

    def mode_input_changed(self, mode_name):
        if mode_name == "Camera":
            self.input_import.setVisible(False)
            self.label_import.setVisible(False)
        else:
            self.input_import.setPlaceholderText(f"Import { mode_name }")
            if not self.label_import.isVisible():
                self.input_import.setVisible(True)
                self.label_import.setVisible(True)

    def on_pause(self):
        self.timer_video.stop()
        self.resume_button.setHidden(False)
        self.pause_button.setHidden(True)
        self.statusBar.showMessage("Paused!")
        self.statusBar.repaint()

    def on_resume(self):
        self.timer_video.start(60)
        self.pause_button.setHidden(False)
        self.resume_button.setHidden(True)
        self.statusBar.showMessage("Resumed!")
        self.statusBar.repaint()

    def on_stop(self):
        self.timer_video.stop()
        self.cap.release()
        self.out.release()
        self.hide_all_buttons()
        self.save_button.setHidden(False)
        self.cancel_button.setHidden(False)
        self.statusBar.showMessage("Stopped!")
        self.statusBar.repaint()

    def resize_img(self, img):
        resized_image = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        resized_image = cv2.resize(
            resized_image, (640, 480), interpolation=cv2.INTER_AREA
        )
        preview_image = QtGui.QImage(
            resized_image.data,
            resized_image.shape[1],
            resized_image.shape[0],
            QtGui.QImage.Format.Format_RGB32,
        ).copy()
        return QtGui.QPixmap.fromImage(preview_image)

    def import_file_open(self):
        if self.input_mode.currentText() == "Image":
            self.img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "IMAGE", "", "Image (*.jpg *.jpeg *.png);;All Files(*)"
            )
            # self.input_image.clear()
            if self.img_name:
                try:
                    _preview_image = cv2.imread(self.img_name)
                    resized_image = self.resize_img(_preview_image)
                    if resized_image.isNull():
                        self.statusBar.showMessage("Failed to render preview image.")
                        return
                    self.input_image.setPixmap(resized_image)
                except:
                    self.statusBar.showMessage("Failed to render preview image.")
                dest = QtCore.QDir(self.img_name)
                self.input_import.setText(QtCore.QDir.fromNativeSeparators(dest.path()))
            else:
                self.input_import.clear()
                self.input_image.setPixmap(
                    QtGui.QPixmap.fromImage(self.background_image)
                )
        else:
            self.video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "VIDEO", "", "Video (*.mp4 *avi);;All Files(*)"
            )
            if self.video_name:
                try:
                    self.cap.open(self.video_name)
                    flag, _preview_image = self.cap.read()
                    resized_image = self.resize_img(_preview_image)
                    if resized_image.isNull():
                        self.statusBar.showMessage("Failed to render preview image.")
                        return
                    self.input_image.setPixmap(resized_image)
                except:
                    self.statusBar.showMessage("Failed to render preview image.")
                dest = QtCore.QDir(self.video_name)
                self.input_import.setText(QtCore.QDir.fromNativeSeparators(dest.path()))
            else:
                self.input_import.clear()
                self.input_image.setPixmap(
                    QtGui.QPixmap.fromImage(self.background_image)
                )

    def init_yolov5_defaults(self):
        self.weights = "weights/yolov5s.pt"
        self.device = select_device("")
        self.augment = None
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.img_size = 640
        self.half = self.device.type != "cpu"

    def on_detect(self):
        mode = self.input_mode.currentText()
        self.statusBar.showMessage(f"{mode} Detection Running. Getting Model ready.")
        self.statusBar.repaint()
        self.classes = [
            idx for idx, val in enumerate(self.class_checkboxes) if val.isChecked()
        ]
        self.model = DetectMultiBackend(
            [self.file_models[self.input_detection_model.currentIndex()].absolute()],
            device=self.device,
        )
        stride, self.names, pt, jit, onnx, engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )
        self.imgsz = check_img_size(self.img_size, s=stride)  # check img_size
        self.half &= (
            pt or engine
        ) and self.device.type != "cpu"  # half precision only supported by PyTorch on CUDA
        if pt:
            self.model.model.half() if self.half else self.model.model.float()
        self.colors = [
            [random.randint(0, 255) for _ in range(3)] for _ in str(self.names)
        ]
        self.hide_all_buttons()
        if mode == "Camera":
            cudnn.benchmark = True
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Warning",
                    "Camera not Found.",
                    buttons=QtWidgets.QMessageBox.StandardButton.Ok,
                    defaultButton=QtWidgets.QMessageBox.StandardButton.Ok,
                )
            else:
                self.pause_button.setHidden(False)
                self.stop_button.setHidden(False)
                self.out = cv2.VideoWriter(
                    "result.avi",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,
                    (int(self.cap.get(3)), int(self.cap.get(4))),
                )
                self.timer_video.start(60)
        elif mode == "Image":
            if not self.input_import.text().endswith(tuple(IMG_FORMATS)):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error",
                    "Please Import an Image.",
                    buttons=QtWidgets.QMessageBox.StandardButton.Ok,
                    defaultButton=QtWidgets.QMessageBox.StandardButton.Ok,
                )
                return
            self.save_button.setHidden(False)
            self.cancel_button.setHidden(False)
            self.before_button.setHidden(False)
            self.after_button.setHidden(False)
            img = cv2.imread(self.img_name)
            showimg = img
            with torch.no_grad():
                dt = [0.0, 0.0, 0.0]
                t1 = time_sync()
                img = letterbox(img, new_shape=self.img_size)[0]
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                # img = img.float()
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                t2 = time_sync()
                dt[0] += t2 - t1
                pred = self.model(img, augment=self.augment)

                t3 = time_sync()
                dt[1] += t3 - t2
                # Apply NMS
                pred = non_max_suppression(
                    pred,
                    self.conf_thres,
                    self.iou_thres,
                    classes=self.classes,
                    agnostic=None,
                )

                dt[2] += time_sync() - t3
                # Process detections

                self.statusBar.showMessage(f"Average FPS Detected %.1fms" % (t3 - t2))
                self.statusBar.repaint()
                annotator = Annotator(showimg, line_width=3, example=str(self.names))
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], showimg.shape
                        ).round()
                        # Write results
                        for idx, c in enumerate(det[:, -1].unique()):
                            n = (det[:, -1] == c).sum()  # detections per class

                            getattr(
                                self,
                                f"occ_class_{self.names.index(self.names[int(c)])}",
                            ).setText(f"{ n }")

                        for *xyxy, conf, cls in reversed(det):
                            label = "%s %.2f" % (self.names[int(cls)], conf)
                            annotator.box_label(
                                xyxy,
                                label=label,
                                color=self.colors[int(cls)],
                            )

            cv2.imwrite("prediction.jpg", showimg)
            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
            self.result = cv2.resize(
                self.result, (640, 480), interpolation=cv2.INTER_AREA
            )
            self.QtImg = QtGui.QImage(
                self.result.data,
                self.result.shape[1],
                self.result.shape[0],
                QtGui.QImage.Format.Format_RGB32,
            )
        else:
            if not self.input_import.text().endswith(tuple(VID_FORMATS)):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error",
                    "Please Import a Video.",
                    buttons=QtWidgets.QMessageBox.StandardButton.Ok,
                    defaultButton=QtWidgets.QMessageBox.StandardButton.Ok,
                )
                return
            self.pause_button.setHidden(False)
            self.stop_button.setHidden(False)
            flag = self.cap.open(self.video_name)
            self.out = cv2.VideoWriter(
                "result.avi",
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (int(self.cap.get(3)), int(self.cap.get(4))),
            )
            self.timer_video.start(60)

    def show_video_frame(self):
        name_list = []

        flag, img = self.cap.read()
        if img is not None:
            showimg = img
            with torch.no_grad():
                dt = [0.0, 0.0, 0.0]
                t1 = time_sync()
                img = letterbox(img, new_shape=self.img_size)[0]
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                # img = img.float()
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                t2 = time_sync()
                dt[0] += t2 - t1
                try:
                    pred = self.model(img, augment=self.augment)
                except:
                    pred = None
                if pred == None:
                    self.statusBar.showMessage("Error Using model!")
                    return

                t3 = time_sync()
                dt[1] += t3 - t2
                # Apply NMS
                pred = non_max_suppression(
                    pred,
                    self.conf_thres,
                    self.iou_thres,
                    classes=self.classes,
                    agnostic=None,
                )
                dt[2] += time_sync() - t3

                self.statusBar.showMessage(
                    f"Average FPS Detected %.4f" % (1000 * (t3 - t2))
                )
                self.statusBar.repaint()
                # Process detections
                annotator = Annotator(showimg, line_width=3, example=str(self.names))
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], showimg.shape
                        ).round()
                        # Write results
                        for idx, c in enumerate(det[:, -1].unique()):
                            n = (det[:, -1] == c).sum()  # detections per class
                            s = f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            # self.class_occurences[
                            #     self.names.index(self.names[int(c)])
                            # ].setText(n)
                            getattr(
                                self,
                                f"occ_class_{self.names.index(self.names[int(c)])}",
                            ).setText(f"{ n }")

                        for *xyxy, conf, cls in reversed(det):
                            label = "%s %.2f" % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            annotator.box_label(
                                xyxy,
                                label=label,
                                color=self.colors[int(cls)],
                            )

            self.out.write(showimg)
            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(
                self.result.data,
                self.result.shape[1],
                self.result.shape[0],
                QtGui.QImage.Format.Format_RGB888,
            )
            self.input_image.setPixmap(QtGui.QPixmap.fromImage(showImage))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    qt_app = TransformerApp()
    qt_app.show()

    sys.exit(app.exec())
