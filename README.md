# read_segd

SEG D bindings to ObsPy

You must have [ObsPy](https://obspy.org) installed to use this package.

## Usage

### From command line

Getting help:

```bash
python read_segd.py -h
```

Converting to miniSEED:

```bash
python read_segd.py -f MSEED file.segd
```

Converting to SAC:

```bash
python read_segd.py -f SAC file.segd
```

### From Python

```python
from read_segd import read_segd
st = read_segd('file.segd')
```
