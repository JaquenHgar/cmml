# cmml

Simple utilities for Python machine learning - things you often need to do, are easy to write, but still annoying.


## Usage in Google Colab

There is probably a better way than to copy the data all the time, but this
works. What does not work is using `!pip install cmml/`. It does not find the
plotting module in the cmml package. It works locally, but not in colab.

```none
!rm -r cmml
!git clone -q https://github.com/JaquenHgar/cmml.git

import importlib.util
spec = importlib.util.spec_from_file_location("cmml.plotting", "cmml/cmml/plotting.py")
cplot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cplot)
```