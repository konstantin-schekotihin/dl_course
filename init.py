from IPython.display import HTML, display, Latex
from ipywidgets import interact, interactive, fixed, interact_manual, widgets

import numpy as np
import pandas as pd
import math 
import torch, torch.nn as nn, torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

# Resolve repository root and ensure course modules are in sys.path
_current_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
_repo_root = _current_dir
# If current_dir is a module subdirectory (e.g. 01_Introduction), repo root is its parent
if os.path.basename(_current_dir) in ['01_Introduction', '02_ML', '03_DL', '04_RL']:
    _repo_root = os.path.dirname(_current_dir)

for _sub in ['', '01_Introduction', '02_ML', '03_DL', '04_RL']:
    _p = os.path.abspath(os.path.join(_repo_root, _sub))
    if _p not in sys.path:
        sys.path.append(_p)

# Global styling configuration
sns.set(style="whitegrid", palette="deep")
plt.rc('figure', figsize = (10,10))
plt.rc('font', size=16)
plt.rc('figure', titlesize=20)
plt.rc('axes', labelsize=16)
plt.rc('ytick', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('legend', fontsize=14)

# Load custom CSS styles (supports local relative paths, repo root, rise.css, and inline fallback for Colab)
_custom_css_paths = [
    './rise.css',
    '../rise.css',
    '../custom.html',
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

# Course LaTeX macros
display(Latex("""$$
\\def\\rvar#1{\\mathrm{#1}}
\\def\\rvec#1{\\mathbf{#1}}
\\def\\vec#1{\\boldsymbol{#1}}
\\def\\tens#1{\\boldsymbol{\\mathsf{#1}}}
\\def\\tensel#1{\\mathsf{#1}}
\\def\\st#1{\\mathcal{#1}}
\\def\\diag#1{\\mathrm{diag}(\\vec{#1})}
$$"""))

# Colab-specific setup (quietly fetch NLTK data if in Colab environment)
if 'google.colab' in sys.modules:
    try:
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
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
