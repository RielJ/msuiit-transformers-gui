<div align="center">

</div>

## <div align="center">Documentation</div>

A GUI application for Distribution Line Detection. It detects Powerlines, Components (Transformer Tank, HV Bushing, LV Bushing, Arrester, Radiator Fins and Cutout Fuse) and Obstructions on a distribution line.

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0 && Python < 3.10.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/RielJ/msuiit-transformers-gui/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):

```bash
# Clone the repository
$ git clone https://github.com/RielJ/msuiit-transformers-gui.git transformers-gui
$ cd transformers-gui

# Optional: Create a python virtual environment
$ python3 -m venv virtualenv
$ source virtualenv/bin/activate
# ./virtualenv/bin/activate -> Windows

# Install Packages.
$ pip install -r requirements.txt
```

</details>
<details open>
<summary>Adding Weights. (required)</summary>

Application needs different models resided in `/weights/*` folder before it can run.


</details>

<details open>
<summary>Running the application</summary>

Detection using YOLOv5. 

```bash
# Optional: If packages is installed in a virtual environment
$ source virtualenv/bin/activate
# ./virtualenv/bin/activate -> Windows

$ python main.py
```

</details>
