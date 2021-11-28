import random
import sys

import cv2
import numpy as np
import torch
from PyQt6 import QtCore, QtGui, QtWidgets

import ui_interface
from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.datasets import IMG_FORMATS, VID_FORMATS
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator
from utils.torch_utils import select_device


class TransformerApp(ui_interface.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(TransformerApp, self).__init__()
        self.timer_video = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.setupUi(self)
        self.showMaximized()
        self.setWindowTitle("Distribution Line Detection")
        self.toggle_import()
        self.init_yolov5_defaults()
        self.init_slots()
        self.button_detection.setPlaceholderText("Import a Model")
        self.class_checkboxes = [
            self.class_0,
            self.class_1,
            self.class_2,
            self.class_3,
            self.class_4,
            self.class_5,
        ]
        # background = QtGui.QPixmap("assets/background.png")
        background_image = QtGui.QImage("assets/background.png")
        background_image.scaledToWidth(640)
        background = QtGui.QPixmap.fromImage(background_image)
        background.scaled(64, 64, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.label.setPixmap(background)

    def init_slots(self):
        self.combo_mode.textActivated.connect(self.mode_input_changed)
        self._open_folder_detection_action = self.button_import.addAction(
            QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon
            ),
            QtWidgets.QLineEdit.ActionPosition.TrailingPosition,
        )
        self._open_folder_detection_action.triggered.connect(self.import_file_open)
        self._open_folder_detection_action = self.button_detection.addAction(
            QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon
            ),
            QtWidgets.QLineEdit.ActionPosition.TrailingPosition,
        )
        self._open_folder_detection_action.triggered.connect(self.import_detection_open)
        self.stop_button.clicked.connect(self.on_stop)
        self.stop_button.setDisabled(True)
        self.detect_button.clicked.connect(self.on_detect)
        self.timer_video.timeout.connect(self.show_video_frame)
        self.actionExit.setShortcut("Ctrl+Q")
        self.actionExit.setStatusTip("Exit application")
        self.actionExit.triggered.connect(QtWidgets.QApplication.quit)

    def mode_input_changed(self, mode_name):
        if mode_name == "Camera":
            self.button_import.setVisible(False)
            self.label_import.setVisible(False)
        else:
            self.button_import.setPlaceholderText(f"Import { mode_name }")
            if not self.label_import.isVisible():
                self.button_import.setVisible(True)
                self.label_import.setVisible(True)

    def on_stop(self):
        self.timer_video.stop()
        self.cap.release()
        self.out.release()
        self.label.clear()
        self.detect_button.setDisabled(False)
        self.stop_button.setDisabled(True)
        # self.list_model.removeRows(0, self.list_model.rowCount())
        self.statusBar.clearMessage()

    def import_detection_open(self):
        self.model_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "MODEL", "weights/", "*.pt"
        )

        if self.model_name:
            dest = QtCore.QDir(self.model_name)
            self.button_detection.setText(QtCore.QDir.fromNativeSeparators(dest.path()))

    def import_file_open(self):
        if self.combo_mode.currentText() == "Image":
            self.img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "IMAGE", "", "Image (*.jpg *.jpeg *.png);;All Files(*)"
            )
            if self.img_name:
                dest = QtCore.QDir(self.img_name)
                self.button_import.setText(
                    QtCore.QDir.fromNativeSeparators(dest.path())
                )
        else:
            self.video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "VIDEO", "", "Video (*.mp4 *avi);;All Files(*)"
            )
            if self.video_name:
                dest = QtCore.QDir(self.video_name)
                self.button_import.setText(
                    QtCore.QDir.fromNativeSeparators(dest.path())
                )

    def init_yolov5_defaults(self):
        self.weights = "weights/yolov5s.pt"
        self.device = select_device("")
        self.augment = None
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.img_size = 640
        self.half = self.device.type != "cpu"

    def toggle_import(self):
        self.button_import.setVisible(not self.button_import.isVisible())
        self.label_import.setVisible(not self.label_import.isVisible())

    def on_detect(self):
        if not self.button_detection.text().endswith(".pt"):
            QtWidgets.QMessageBox.warning(
                self,
                "Error",
                "Please Import a model.",
                buttons=QtWidgets.QMessageBox.StandardButton.Ok,
                defaultButton=QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return
        mode = self.combo_mode.currentText()
        self.statusBar.showMessage(f"{mode} Detection Running. Getting Model ready.")
        self.classes = [
            idx for idx, val in enumerate(self.class_checkboxes) if val.isChecked()
        ]
        self.model = DetectMultiBackend(
            [self.button_detection.text()], device=self.device
        )
        # self.model = attempt_load(
        #     [self.button_detection.text()], map_location=self.device
        # )  # load model
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
        for idx, name in enumerate(self.names):
            item = QtWidgets.QTableWidgetItem(str(idx))
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter),
            self.prediction_table.setItem(idx, 0, item)
            self.prediction_table.setItem(
                idx,
                1,
                QtWidgets.QTableWidgetItem(name),
            )
            item = QtWidgets.QTableWidgetItem(str(0))
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter),
            self.prediction_table.setItem(idx, 2, item)
        self.detect_button.setDisabled(True)
        self.stop_button.setDisabled(False)
        if mode == "Camera":
            self.statusBar.showMessage(f"Opening Camera!")
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
                self.out = cv2.VideoWriter(
                    "prediction.avi",
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    20,
                    (int(self.cap.get(3)), int(self.cap.get(4))),
                )
                self.timer_video.start(60)
        elif mode == "Image":
            self.stop_button.setDisabled(True)
            if not self.button_import.text().endswith(tuple(IMG_FORMATS)):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error",
                    "Please Import an Image.",
                    buttons=QtWidgets.QMessageBox.StandardButton.Ok,
                    defaultButton=QtWidgets.QMessageBox.StandardButton.Ok,
                )
                return
            self.statusBar.showMessage(f"Reading Image")
            img = cv2.imread(self.img_name)
            showimg = img
            with torch.no_grad():
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
                pred = self.model(img, augment=self.augment)

                # Apply NMS
                pred = non_max_suppression(
                    pred,
                    self.conf_thres,
                    self.iou_thres,
                    classes=self.classes,
                    agnostic=None,
                )

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
                            item = QtWidgets.QTableWidgetItem(f"{n}")
                            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter),

                            self.prediction_table.setItem(
                                self.names.index(self.names[int(c)]), 2, item
                            )

                        for *xyxy, conf, cls in reversed(det):
                            label = "%s %.2f" % (self.names[int(cls)], conf)
                            self.statusBar.showMessage(label)
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
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

            self.detect_button.setDisabled(False)
        else:
            if not self.button_import.text().endswith(tuple(VID_FORMATS)):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error",
                    "Please Import a Video.",
                    buttons=QtWidgets.QMessageBox.StandardButton.Ok,
                    defaultButton=QtWidgets.QMessageBox.StandardButton.Ok,
                )
                return
            self.statusBar.showMessage(f"Reading Video")
            flag = self.cap.open(self.video_name)
            self.out = cv2.VideoWriter(
                "results/result.avi",
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
                pred = self.model(img, augment=self.augment)

                # Apply NMS
                pred = non_max_suppression(
                    pred,
                    self.conf_thres,
                    self.iou_thres,
                    classes=self.classes,
                    agnostic=None,
                )

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
                            item = QtWidgets.QTableWidgetItem(f"{n}")
                            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter),
                            self.prediction_table.setItem(
                                self.names.index(self.names[int(c)]), 2, item
                            )

                        for *xyxy, conf, cls in reversed(det):
                            label = "%s %.2f" % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            self.statusBar.showMessage(label)
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
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    qt_app = TransformerApp()
    qt_app.show()

    sys.exit(app.exec())
