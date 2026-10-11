# dl-course

Helper code for the notebooks of the [AAU Machine Learning and Deep Learning course](https://github.com/konstantin-schekotihin/dl_course).
It covers plotting, slide styling and loading course datasets and images.

The package is not a general-purpose library. It only serves the course notebooks.

## Usage in Google Colab

The first cell of every notebook installs the package when it runs in Colab:

```python
import sys
if "google.colab" in sys.modules:
    %pip install -q dl-course
from dl_course import deep_learning as hp
```

## Modules

| Module | Used by |
|---|---|
| `dl_course.shared` | common base: themes, dataset/image loading, shared plots |
| `dl_course.aiml` | AI-ML course |
| `dl_course.foundations` | ML-DL, part 1: mathematical foundations |
| `dl_course.deep_learning` | ML-DL, part 2: deep learning |

## Development and releases

- **Local development:** in the course repository, `uv sync` installs this package in editable mode. Changes take effect after a kernel restart.
- **Releases:** every push to `master` that changes `packages/dl-course/` publishes a new version to PyPI. A GitHub Action (`.github/workflows/publish-dl-course.yml`) does this, using a date-based version number.

## License

MIT (see the `LICENSE` file in the package source).
