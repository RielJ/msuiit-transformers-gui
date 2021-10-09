from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MSU-IIT Transformers")

        mainLayout = QHBoxLayout()
        mainLayout.setSpacing(30)

        col1 = QWidget()
        col1_layout = QVBoxLayout()
        col1_layout.setSpacing(30)
        col1_layout.addWidget(self.video_panel_ui())
        col1_layout.addWidget(self.prediction_panel_ui())
        col1.setLayout(col1_layout)

        col2 = QWidget()
        col2_layout = QVBoxLayout()
        col2_layout.addWidget(self.form_panel_ui())
        col2.setLayout(col2_layout)

        mainLayout.addWidget(col1)
        mainLayout.addWidget(col2)

        self.setLayout(mainLayout)

    def video_panel_ui(self):
        # video_panel = QWidget()
        # return video_panel
        video_label = QLabel("Video Panel")
        return video_label

    def prediction_panel_ui(self):
        # prediction_panel = QWidget()
        # return prediction_panel
        prediction_label = QLabel("Prediction Panel")
        return prediction_label

    def form_panel_ui(self):
        form_panel = QWidget()
        form_layout = QFormLayout()

        action_button_box = QWidget()
        button_detect = QPushButton("Detect")
        button_save = QPushButton("Save")
        form_layout.addRow(button_detect, button_save)

        form_panel.setLayout(form_layout)
        form_layout.addWidget(action_button_box)
        return form_panel
