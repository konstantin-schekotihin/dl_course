// dl-course: keep every formula on one line in the classic Jupyter Notebook and in RISE.
// The classic Notebook turns on MathJax's automatic line breaking. MathJax then breaks formulas in
// Markdown table cells at the narrow width the cell has while it is typeset. Colab, VS Code and
// JupyterLab do not use this option, so they need no fix.
define(function () {
    "use strict";

    function load_ipython_extension() {
        if (!window.MathJax || !window.MathJax.Hub) {
            return;
        }
        MathJax.Hub.Config({"HTML-CSS": {linebreaks: {automatic: false}}});
        MathJax.Hub.Queue(["Rerender", MathJax.Hub]);   // re-typeset the cells rendered so far
    }

    return {load_ipython_extension: load_ipython_extension};
});
