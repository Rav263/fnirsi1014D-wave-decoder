# FNIRSI 1014D Oscilloscope Trace Decoder

Decodes proprietary `.wav` trace files saved by the **FNIRSI 1014D** oscilloscope and exports them as:

- **CSV** — time (ns) + voltage (mV) + raw ADC values
- **PNG** — waveform plot with oscilloscope-style dark theme
- **Tektronix-compatible bundle** (optional) — per-channel CSV + BMP image in `ALLxxxx/` directory structure

Supports single-channel and dual-channel traces.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic — export to CSV + PNG

```bash
python fnirsi_decoder.py trace1.wav trace2.wav
```

Output files are saved next to the input files (`trace1.csv`, `trace1.png`, etc.).

### Specify output directory

```bash
python fnirsi_decoder.py test_oscil/*.wav -o output
```

### Tektronix-compatible export (`-t`)

```bash
python fnirsi_decoder.py test_oscil/4.wav test_oscil/7.wav -o output -t
```

This additionally creates a Tektronix TDS2012C-compatible directory structure:

```
output/
├── 4.csv                  # standard CSV
├── 4.png                  # standard PNG plot
├── ALL0004/
│   ├── F0004CH1.CSV       # CH1 in Tektronix CSV format
│   └── F0004TEK.BMP       # waveform plot as BMP
├── 7.csv
├── 7.png
└── ALL0007/
    ├── F0007CH1.CSV
    ├── F0007CH2.CSV       # CH2 (dual-channel traces only)
    └── F0007TEK.BMP
```

The Tektronix CSV format includes a standard 18-row header (`Record Length`, `Sample Interval`, `Vertical Scale`, `Horizontal Scale`, etc.) with time in seconds and voltage in volts — compatible with tools that import TDS2012C data.

### Command-line options

| Option | Description |
|---|---|
| `FILE ...` | One or more `.wav` trace files to decode |
| `-o DIR` | Output directory (default: same as input file) |
| `-t` | Also export in Tektronix-compatible format |

### Example output

```
==================================================
  File:       4.wav
  Timebase:   500ns/div  (index 25)
  Sample int: 5ns  (1500 samples, total 7.495µs)
  CH1:        Vpp=1313mV, GND offset=202, ADC range=[139-267]
  CH2:        disabled
  CSV:        output/4.csv
  PNG:        output/4.png
```

## FNIRSI 1014D WAV file format

All trace files are exactly **15000 bytes** with the following layout:

| Offset (bytes) | Size | Content |
|---|---|---|
| 0–999 | 1000 B | Header (oscilloscope settings, measurements) |
| 1000–3999 | 3000 B | CH1 data — 1500 samples, uint16 LE |
| 4000–6999 | 3000 B | CH2 data — 1500 samples, uint16 LE (zeros if single channel) |
| 7000–14999 | 8000 B | Extra data |

### Key header fields (uint16 LE)

| Byte offset | Description |
|---|---|
| `0x0C` | CH2 enabled (0 = off, 1 = on) |
| `0x16` | Timebase index (higher = faster; 25 = 500ns/div) |
| `0x52` | CH1 GND offset (ADC value for 0V reference) |
| `0x54` | CH2 GND offset |
| `0xD2` | CH1 Vpp in mV |
| `0x102` | CH2 Vpp in mV |

### Timebase index mapping

| Index | Time/div | Index | Time/div |
|---|---|---|---|
| 25 | 500 ns | 17 | 200 µs |
| 24 | 1 µs | 16 | 500 µs |
| 23 | 2 µs | 15 | 1 ms |
| 22 | 5 µs | 14 | 2 ms |
| 21 | 10 µs | 13 | 5 ms |
| 20 | 20 µs | 12 | 10 ms |
| 19 | 50 µs | 11 | 20 ms |
| 18 | 100 µs | 10 | 50 ms |

## License

MIT
