#!/usr/bin/env python3
"""
FNIRSI 1014D Oscilloscope WAV Trace Decoder

Decodes proprietary .wav files saved by FNIRSI 1014D oscilloscope
and exports them as .csv (data) and .png (plot).

File format (15000 bytes total):
  - Header:    bytes 0-999    (500 uint16 LE values)
  - CH1 data:  bytes 1000-3999 (1500 uint16 LE samples)
  - CH2 data:  bytes 4000-6999 (1500 uint16 LE samples, zeros if single channel)
  - Extra:     bytes 7000+     (additional data, ignored)

Header fields (uint16 LE at byte offset):
  0x0C  (idx  6):  CH2 enabled flag (0=off, 1=on)
  0x16  (idx 11):  Timebase index
  0x52  (idx 41):  CH1 GND offset (ADC value for 0V)
  0x54  (idx 42):  CH2 GND offset
  0xD2  (idx 105): CH1 Vpp in mV
  0x102 (idx 129): CH2 Vpp in mV
"""

import struct
import csv
import sys
import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io

# --- Constants ---

FILE_SIZE = 15000
SAMPLES_PER_CHANNEL = 1500
SAMPLES_PER_DIV = 100   # 100 samples per horizontal division
N_HDIV = 15             # number of horizontal divisions

# Data offsets (uint16 index)
CH1_START = 500   # byte 1000
CH2_START = 2000  # byte 4000

# Timebase lookup: index → nanoseconds per division
# 1-2-5 sequence, higher index = faster timebase
# Empirically verified: tb=25 → 500ns/div, tb=24 → 1µs/div
TIMEBASE_NS_PER_DIV = {
    0: 100_000_000_000,  # 100s
    1:  50_000_000_000,  # 50s
    2:  20_000_000_000,  # 20s
    3:  10_000_000_000,  # 10s
    4:   5_000_000_000,  # 5s
    5:   2_000_000_000,  # 2s
    6:   1_000_000_000,  # 1s
    7:     500_000_000,  # 500ms
    8:     200_000_000,  # 200ms
    9:     100_000_000,  # 100ms
   10:      50_000_000,  # 50ms
   11:      20_000_000,  # 20ms
   12:      10_000_000,  # 10ms
   13:       5_000_000,  # 5ms
   14:       2_000_000,  # 2ms
   15:       1_000_000,  # 1ms
   16:         500_000,  # 500µs
   17:         200_000,  # 200µs
   18:         100_000,  # 100µs
   19:          50_000,  # 50µs
   20:          20_000,  # 20µs
   21:          10_000,  # 10µs
   22:           5_000,  # 5µs
   23:           2_000,  # 2µs
   24:           1_000,  # 1µs
   25:             500,  # 500ns
}


def format_time(ns):
    """Format nanoseconds into a human-readable string."""
    if ns >= 1e9:
        return f"{ns/1e9:.6g}s"
    elif ns >= 1e6:
        return f"{ns/1e6:.6g}ms"
    elif ns >= 1e3:
        return f"{ns/1e3:.6g}µs"
    else:
        return f"{ns:.6g}ns"


def format_time_per_div(ns_per_div):
    """Format timebase as time/div string."""
    return format_time(ns_per_div) + "/div"


def parse_trace(filepath):
    """
    Parse an FNIRSI 1014D .wav trace file.

    Returns a dict with:
      ch1_raw, ch2_raw: raw ADC values (list of int)
      ch1_mV, ch2_mV:   voltage in millivolts (list of float)
      time_ns:           time axis in nanoseconds (list of float)
      ch2_enabled:       bool
      timebase_idx:      int
      ns_per_div:        float
      ch1_vpp_mV, ch2_vpp_mV: int
      ch1_gnd, ch2_gnd:  int (GND ADC offsets)
      sample_interval_ns: float
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) != FILE_SIZE:
        raise ValueError(
            f"Invalid file size: expected {FILE_SIZE} bytes, got {len(data)}"
        )

    vals = struct.unpack('<' + 'H' * (FILE_SIZE // 2), data)

    # --- Parse header ---
    ch2_enabled = vals[6] != 0
    timebase_idx = vals[11]
    ch1_gnd = vals[41]
    ch2_gnd = vals[42]
    ch1_vpp_mV = vals[105]
    ch2_vpp_mV = vals[129] if ch2_enabled else 0

    # --- Extract raw samples ---
    ch1_raw = list(vals[CH1_START:CH1_START + SAMPLES_PER_CHANNEL])
    ch2_raw = (
        list(vals[CH2_START:CH2_START + SAMPLES_PER_CHANNEL])
        if ch2_enabled else None
    )

    # --- Time axis ---
    ns_per_div = TIMEBASE_NS_PER_DIV.get(timebase_idx)
    if ns_per_div is None:
        print(f"  Warning: unknown timebase index {timebase_idx}, "
              f"defaulting to 1µs/div", file=sys.stderr)
        ns_per_div = 1000  # 1µs/div fallback

    sample_interval_ns = ns_per_div / SAMPLES_PER_DIV
    time_ns = [i * sample_interval_ns for i in range(SAMPLES_PER_CHANNEL)]

    # --- Voltage conversion ---
    def adc_to_mV(raw_samples, gnd_offset, vpp_mV):
        lo = min(raw_samples)
        hi = max(raw_samples)
        adc_range = hi - lo
        if adc_range > 0 and vpp_mV > 0:
            mV_per_count = vpp_mV / adc_range
        else:
            # Flat signal or no Vpp info — assume 10 mV/count (≈500mV/div)
            mV_per_count = 10.0
        return [round((v - gnd_offset) * mV_per_count, 2) for v in raw_samples]

    ch1_mV = adc_to_mV(ch1_raw, ch1_gnd, ch1_vpp_mV)
    ch2_mV = adc_to_mV(ch2_raw, ch2_gnd, ch2_vpp_mV) if ch2_raw else None

    return {
        'ch1_raw': ch1_raw,
        'ch2_raw': ch2_raw,
        'ch1_mV': ch1_mV,
        'ch2_mV': ch2_mV,
        'time_ns': time_ns,
        'ch2_enabled': ch2_enabled,
        'timebase_idx': timebase_idx,
        'ns_per_div': ns_per_div,
        'ch1_vpp_mV': ch1_vpp_mV,
        'ch2_vpp_mV': ch2_vpp_mV,
        'ch1_gnd': ch1_gnd,
        'ch2_gnd': ch2_gnd,
        'sample_interval_ns': sample_interval_ns,
    }


def save_csv(trace, output_path):
    """Export trace data to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        if trace['ch2_enabled']:
            writer.writerow([
                'time_ns', 'ch1_mV', 'ch2_mV', 'ch1_raw', 'ch2_raw'
            ])
            for i in range(SAMPLES_PER_CHANNEL):
                writer.writerow([
                    f"{trace['time_ns'][i]:.2f}",
                    f"{trace['ch1_mV'][i]:.2f}",
                    f"{trace['ch2_mV'][i]:.2f}",
                    trace['ch1_raw'][i],
                    trace['ch2_raw'][i],
                ])
        else:
            writer.writerow(['time_ns', 'ch1_mV', 'ch1_raw'])
            for i in range(SAMPLES_PER_CHANNEL):
                writer.writerow([
                    f"{trace['time_ns'][i]:.2f}",
                    f"{trace['ch1_mV'][i]:.2f}",
                    trace['ch1_raw'][i],
                ])


def choose_time_units(max_ns):
    """Choose appropriate time units for the plot axis."""
    if max_ns >= 1e9:
        return 1e9, 's'
    elif max_ns >= 1e6:
        return 1e6, 'ms'
    elif max_ns >= 1e3:
        return 1e3, 'µs'
    else:
        return 1, 'ns'


def save_png(trace, output_path, title=''):
    """Generate a PNG plot of the trace."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Time units
    factor, unit = choose_time_units(trace['time_ns'][-1])
    time_plot = [t / factor for t in trace['time_ns']]

    # Timebase label
    tb_label = format_time_per_div(trace['ns_per_div'])

    # Plot channels
    ch1_label = f"CH1 (Vpp={trace['ch1_vpp_mV']}mV)"
    ax.plot(time_plot, trace['ch1_mV'],
            label=ch1_label, color='#FFD700', linewidth=0.8)

    if trace['ch2_enabled']:
        ch2_label = f"CH2 (Vpp={trace['ch2_vpp_mV']}mV)"
        ax.plot(time_plot, trace['ch2_mV'],
                label=ch2_label, color='#00FFFF', linewidth=0.8)

    # Styling (dark oscilloscope theme)
    ax.set_xlabel(f'Time ({unit})', color='white')
    ax.set_ylabel('Voltage (mV)', color='white')
    ax.set_title(
        (title or os.path.basename(output_path)) + f'  [{tb_label}]',
        color='white', fontsize=13
    )
    ax.legend(
        facecolor='#16213e', edgecolor='#555',
        labelcolor='white', fontsize=10
    )
    ax.grid(True, alpha=0.25, color='#446688')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0f0f23')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')

    # Zero line
    ax.axhline(y=0, color='#666', linewidth=0.5, linestyle='--')

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def save_plot(trace, output_path, title='', fmt='png'):
    """Generate a plot of the trace in the given format (png or bmp)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    factor, unit = choose_time_units(trace['time_ns'][-1])
    time_plot = [t / factor for t in trace['time_ns']]
    tb_label = format_time_per_div(trace['ns_per_div'])

    ch1_label = f"CH1 (Vpp={trace['ch1_vpp_mV']}mV)"
    ax.plot(time_plot, trace['ch1_mV'],
            label=ch1_label, color='#FFD700', linewidth=0.8)

    if trace['ch2_enabled']:
        ch2_label = f"CH2 (Vpp={trace['ch2_vpp_mV']}mV)"
        ax.plot(time_plot, trace['ch2_mV'],
                label=ch2_label, color='#00FFFF', linewidth=0.8)

    ax.set_xlabel(f'Time ({unit})', color='white')
    ax.set_ylabel('Voltage (mV)', color='white')
    ax.set_title(
        (title or os.path.basename(output_path)) + f'  [{tb_label}]',
        color='white', fontsize=13
    )
    ax.legend(
        facecolor='#16213e', edgecolor='#555',
        labelcolor='white', fontsize=10
    )
    ax.grid(True, alpha=0.25, color='#446688')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0f0f23')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.axhline(y=0, color='#666', linewidth=0.5, linestyle='--')

    if fmt == 'bmp':
        buf = io.BytesIO()
        fig.savefig(buf, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), format='png')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img.save(output_path, format='BMP')
        buf.close()
    else:
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), format=fmt)
        plt.close(fig)


def save_tek_csv(trace, output_path, channel='CH1'):
    """
    Export a single channel in Tektronix-compatible CSV format.

    Format matches TDS2012C CSV output:
      - First 18 rows: header fields in cols 1-2 + data in cols 4-5
      - Remaining rows: data only in cols 4-5 (prefixed with ,,,)
    """
    n_samples = len(trace['time_ns'])
    sample_interval_s = trace['sample_interval_ns'] * 1e-9
    ns_per_div = trace['ns_per_div']
    h_scale_s = ns_per_div * 1e-9

    if channel == 'CH2' and trace['ch2_enabled']:
        mV_data = trace['ch2_mV']
        vpp_mV = trace['ch2_vpp_mV']
        source = 'CH2'
    else:
        mV_data = trace['ch1_mV']
        vpp_mV = trace['ch1_vpp_mV']
        source = 'CH1'

    # Voltage in V
    v_data = [mv / 1000.0 for mv in mV_data]

    # Time axis in seconds, centered around trigger (midpoint)
    trigger_sample = n_samples // 2
    time_s = [(i - trigger_sample) * sample_interval_s for i in range(n_samples)]

    # Estimate vertical scale from Vpp
    v_scale = vpp_mV / 1000.0 / 4.0  # ~4 divisions for Vpp
    # Round to nearest standard V/div value
    std_vdiv = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    v_scale = min(std_vdiv, key=lambda x: abs(x - v_scale))

    # Header rows (Tektronix TDS format)
    header_rows = [
        (f'Record Length,{n_samples:.6e}',),
        (f'Sample Interval,{sample_interval_s:.6e}',),
        (f'Trigger Point,{trigger_sample:.12e}',),
        ('',),
        ('',),
        ('',),
        (f'Source,{source}',),
        ('Vertical Units,V',),
        (f'Vertical Scale,{v_scale:.6e}',),
        ('Vertical Offset,0.000000e+00',),
        ('Horizontal Units,s',),
        (f'Horizontal Scale,{h_scale_s:.6e}',),
        ('Pt Fmt,Y',),
        ('Yzero,0.000000e+00',),
        ('Probe Atten,1.000000e+00',),
        ('Model Number,FNIRSI 1014D',),
        ('Serial Number,',),
        ('Firmware Version,',),
    ]

    with open(output_path, 'w', newline='') as f:
        for i in range(n_samples):
            t_val = time_s[i]
            v_val = v_data[i]

            if i < len(header_rows):
                hdr = header_rows[i][0]
                t_str = f'  {t_val:+.12f}'
                v_str = f'  {v_val:+.5f}'
                if hdr:
                    f.write(f'{hdr},,{t_str},{v_str},\n')
                else:
                    f.write(f',,,{t_str},{v_str},\n')
            else:
                t_str = f'{t_val:+015.12f}'
                v_str = f'  {v_val:+.5f}'
                f.write(f',,,{t_str},{v_str},\n')


def save_tek_bundle(trace, out_dir, name, title=''):
    """
    Save trace in Tektronix-compatible directory structure:
      ALL{name}/
        F{name}CH1.CSV   — CH1 data
        F{name}CH2.CSV   — CH2 data (if dual channel)
        F{name}TEK.BMP   — waveform plot
    """
    bundle_dir = os.path.join(out_dir, f'ALL{name}')
    os.makedirs(bundle_dir, exist_ok=True)

    # CH1 CSV
    ch1_csv = os.path.join(bundle_dir, f'F{name}CH1.CSV')
    save_tek_csv(trace, ch1_csv, channel='CH1')

    # CH2 CSV (if enabled)
    ch2_csv = None
    if trace['ch2_enabled']:
        ch2_csv = os.path.join(bundle_dir, f'F{name}CH2.CSV')
        save_tek_csv(trace, ch2_csv, channel='CH2')

    # BMP plot
    bmp_path = os.path.join(bundle_dir, f'F{name}TEK.BMP')
    save_plot(trace, bmp_path, title=title, fmt='bmp')

    return bundle_dir, ch1_csv, ch2_csv, bmp_path


def print_info(filepath, trace):
    """Print summary info about the trace."""
    name = os.path.basename(filepath)
    tb = format_time_per_div(trace['ns_per_div'])
    si = format_time(trace['sample_interval_ns'])
    total = format_time(trace['time_ns'][-1])

    print(f"  File:       {name}")
    print(f"  Timebase:   {tb}  (index {trace['timebase_idx']})")
    print(f"  Sample int: {si}  ({SAMPLES_PER_CHANNEL} samples, total {total})")
    print(f"  CH1:        Vpp={trace['ch1_vpp_mV']}mV, "
          f"GND offset={trace['ch1_gnd']}, "
          f"ADC range=[{min(trace['ch1_raw'])}-{max(trace['ch1_raw'])}]")
    if trace['ch2_enabled']:
        print(f"  CH2:        Vpp={trace['ch2_vpp_mV']}mV, "
              f"GND offset={trace['ch2_gnd']}, "
              f"ADC range=[{min(trace['ch2_raw'])}-{max(trace['ch2_raw'])}]")
    else:
        print(f"  CH2:        disabled")


def main():
    parser = argparse.ArgumentParser(
        description='FNIRSI 1014D oscilloscope WAV trace decoder — '
                    'converts .wav traces to .csv and .png'
    )
    parser.add_argument(
        'files', nargs='+', metavar='FILE',
        help='WAV trace file(s) to decode'
    )
    parser.add_argument(
        '-o', '--output-dir', default=None,
        help='Output directory (default: same directory as input file)'
    )
    parser.add_argument(
        '-t', '--tek', action='store_true',
        help='Also export in Tektronix-compatible format '
             '(ALL{name}/ directory with per-channel CSV + BMP)'
    )
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for filepath in args.files:
        if not os.path.isfile(filepath):
            print(f"ERROR: File not found: {filepath}", file=sys.stderr)
            continue

        basename = os.path.splitext(os.path.basename(filepath))[0]
        out_dir = args.output_dir or os.path.dirname(filepath) or '.'

        try:
            trace = parse_trace(filepath)
        except ValueError as e:
            print(f"ERROR: {filepath}: {e}", file=sys.stderr)
            continue

        csv_path = os.path.join(out_dir, f'{basename}.csv')
        png_path = os.path.join(out_dir, f'{basename}.png')

        save_csv(trace, csv_path)
        save_png(trace, png_path, title=f'FNIRSI 1014D — {basename}')

        print(f"\n{'='*50}")
        print_info(filepath, trace)
        print(f"  CSV:        {csv_path}")
        print(f"  PNG:        {png_path}")

        if args.tek:
            tek_name = basename.zfill(4)
            bundle_dir, ch1_csv, ch2_csv, bmp_path = save_tek_bundle(
                trace, out_dir, tek_name,
                title=f'FNIRSI 1014D — {basename}'
            )
            print(f"  TEK dir:    {bundle_dir}")
            print(f"  TEK CH1:    {ch1_csv}")
            if ch2_csv:
                print(f"  TEK CH2:    {ch2_csv}")
            print(f"  TEK BMP:    {bmp_path}")


if __name__ == '__main__':
    main()
