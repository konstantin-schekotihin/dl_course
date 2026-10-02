from IPython.display import HTML, display, Latex
from ipywidgets import interact, interactive, fixed, interact_manual, widgets

import numpy as np
import pandas as pd
import math 
import torch, torch.nn as nn, torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

# Resolve repository root dynamically by walking upwards until pyproject.toml is found
_current_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
_repo_root = _current_dir
while _repo_root != os.path.dirname(_repo_root):
    if os.path.exists(os.path.join(_repo_root, 'pyproject.toml')):
        break
    _repo_root = os.path.dirname(_repo_root)

# Register course modules and shared packages in sys.path
for _sub in [
    '',
    os.path.join('ML-DL', '01_foundations'),
    os.path.join('ML-DL', '02_deep_learning'),
    'AI-ML',
    os.path.join('Extended', '01_reinforcement_learning'),
    'shared',
]:
    _p = os.path.abspath(os.path.join(_repo_root, _sub))
    if _p not in sys.path:
        sys.path.append(_p)

# Canonical asset directories and remote base URLs
DATA_DIR = os.path.join(_repo_root, "shared", "data")
IMAGES_DIR = os.path.join(_repo_root, "shared", "images")
DATA_BASE_URL = os.environ.get(
    "COURSE_DATA_URL",
    "https://raw.githubusercontent.com/konstantin-schekotihin/dl_course/master/shared/data"
)
IMAGE_BASE_URL = os.environ.get(
    "COURSE_IMAGE_URL",
    "https://raw.githubusercontent.com/konstantin-schekotihin/dl_course/master/shared/images"
)

# Global styling configuration
sns.set_theme(
    style="whitegrid",
    palette="deep",
    rc={
        'figure.figsize': (10, 10),
        'font.size': 16,
        'figure.titlesize': 20,
        'axes.labelsize': 16,
        'ytick.labelsize': 14,
        'xtick.labelsize': 14,
        'legend.fontsize': 14,
    }
)

# Load custom CSS styles (supports shared/styles, local relative paths, repo root, and inline fallback for Colab)
_custom_css_paths = [
    os.path.join(_repo_root, 'shared', 'styles', 'rise.css'),
    os.path.join(_repo_root, 'shared', 'styles', 'custom.html'),
    './rise.css',
    '../rise.css',
    '../../rise.css',
    '../custom.html',
    '../../custom.html',
    './custom.html',
    os.path.join(_current_dir, 'rise.css'),
    os.path.join(_repo_root, 'rise.css'),
    os.path.join(_repo_root, 'custom.html'),
    os.path.join(_current_dir, 'custom.html')
]
_css_loaded = False
for _path in _custom_css_paths:
    if os.path.isfile(_path):
        try:
            if _path.endswith('.css'):
                with open(_path, 'r') as _f:
                    display(HTML(f"<style>\n{_f.read()}\n</style>"))
            else:
                display(HTML(filename=_path))
            _css_loaded = True
            break
        except Exception:
            pass

if not _css_loaded:
    display(HTML("""
<style>
.myalert { 
   margin:10pt; 
   border-left: 6px solid darkred;
   background-color:#f5f5f5; 
   color:#0014ff; 
   text-align: center; 
   padding:10px; 
   font-size: 130%;
}
.mycomment {
   font-size: 80%; 
   background-color:#f5f5f5; 
   padding: 5px;
}
.cite {
   font-size: 80%; 
   color:#0000f5; 
   padding: 5px;
}
.mydef{
    margin:10pt; 
    padding-left: 10pt;
    border-left: 6px solid darkblue;
    background-color:whitesmoke; 
}
.reveal .slides section .fragment.step-fade-in-then-out {
    opacity: 0;
    display: none; 
}
.reveal .slides section .fragment.step-fade-in-then-out.current-fragment {
    opacity: 1;
    display: inline; 
}
</style>
"""))
# Colab-specific setup (quietly fetch NLTK data & bridge Colab userdata secrets)
if 'google.colab' in sys.modules:
    try:
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception:
        pass

    try:
        from google.colab import userdata
        if 'WANDB_API_KEY' not in os.environ:
            wandb_key = userdata.get('WANDB_API_KEY')
            if wandb_key:
                os.environ['WANDB_API_KEY'] = wandb_key
        if 'HF_TOKEN' not in os.environ:
            hf_token = userdata.get('HF_TOKEN')
            if hf_token:
                os.environ['HF_TOKEN'] = hf_token
    except Exception:
        pass

# Plot vectors of a 2D tensor (matrix)
def plot2d(tensor):
    plt.figure(figsize=(5,5))
    o = torch.zeros(tensor.shape)
    print('Plotting tensor with dim: {} and shape: {}'.format(tensor.dim(),tensor.shape))
    plt.quiver(*o, *tensor, angles='xy', scale_units='xy', scale=1, color=['r','g','b'])
    mx = torch.max(tensor) if torch.max(tensor) > abs(torch.min(tensor)) else abs(torch.min(tensor))
    plt.xlim(-mx, mx)
    plt.ylim(-mx, mx)
    plt.show();
