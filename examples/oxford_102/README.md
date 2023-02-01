# Oxford Flowers 102 Classification

This directory contains an example showing how to use `bocas` to train four
models on the Oxford Flowers 102 dataset, generate plots, charts, and tables from the
results  of the four experiments, and render those experiments into a pandoc style markdown
project.  The process for a `Latex` project is identical - but does require some more
use of the `\input{}` command sequence.

```
-m bocas.launch --config config.py --task run.py
```
