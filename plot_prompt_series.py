"""Harry King 15/07/2025

This code plots prompt aberration as a function of shot number as a series of
png images. It expects a particular directory structure:

    intdir
    ├── 00 pre shot
    │   └── average over 10
    ├── 01
    ├── 02
    ...
    └── 24

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.optimize import curve_fit
import xml.etree.ElementTree as ET
import matplotlib.dates as mdates
from tqdm import tqdm
import re
import os
from datetime import datetime

def main():
    make_png_series(
        intdir = r"S:\Amplifier testing 2024\20250625\Series 2",
        removetilt = True,
        cbar_mode = "percentiles"
    )

def make_png_series(intdir:str, removetilt:bool=True, terms=None,
              cbar_mode:str="percentiles", cbar_values:tuple=None):
    """Run this function first to generate series of .png files.
    
    Args:
        * intdir: Folder path containing .int and .mcd files
        * removetilt: Switch for removing tip/tilt from wavefront maps.
        * terms: Indices of zernike or legendre coefficients to plot in graph.
          It is recommended to use the default unless fine control is needed.
        * cbar_mode: Determines how colorbar will be scaled - 4 options:
            - "variable" - autoscale for every frame
            - "minmax" - fixed scale based on global min/max wavefront
            - "manual" - cbar_values must be manually specified
            - "percentiles" - cbar_values defaults to (0.1, 99.9) but can
              also be manually specified
        * cbar_values: Behaviour depends on cbar_mode (see above)
    """
    parent = Path.home() / "Documents" / "py_vid"
    safe_mkdir(parent)
    safe_mkdir(parent / "out")
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = parent / "out" / now
    outdir.mkdir()
    (outdir / "img").mkdir()
    temp = get_data(intdir)
    times, wavefronts, coeffs, coefftype, shotnumbers, background = temp
    if terms == None:
        terms = get_default_terms(removetilt, terms, coefftype)
    message = ""
    if removetilt:
        message += "filtering tip/tilt, "
    message += "Computing RMS"
    print(message)
    RMS = []
    for i, wavefront in tqdm(enumerate(wavefronts)):
        wavefronts[i] = wavefront - background
        if removetilt:
            wavefronts[i] = remove_tilt(wavefront)
        RMS.append(np.sqrt(np.nanmean(wavefront**2)))
    RMS = np.array(RMS)
    minval, maxval = get_cbar_scale(wavefronts, cbar_mode, cbar_values)
    print("Plotting")
    plotframes(wavefronts, times, minval, maxval, outdir, intdir, coeffs, RMS,
               removetilt, coefftype, terms, shotnumbers)
    os.startfile(outdir)

def get_default_terms(removetilt, terms, coefftype):
    if coefftype == "legendre":
        terms = np.arange(0, 9)
    elif coefftype == "zernike":
        terms = np.arange(0, 10)
    else:
        raise ValueError(f"Unknown coeff type {coefftype}. "
                         "Must be 'legendre' or 'zernike'")
    if removetilt:
        terms = terms[2:]
    return terms

def safe_mkdir(path:Path):
    if not(path.exists()):
        path.mkdir()

def get_data(intdir):
    print(f"Loading .int, .mcd files from {intdir}")
    if not(Path(intdir).exists()):
        raise FileNotFoundError("Cannot find source folder. "
                                "Perhaps wake up network drive?")
    subdirs = []
    times = []
    wavefronts = []
    coeffs = []
    for file in Path(intdir).glob("*"):
        if file.is_dir():
            subdirs.append(file)
    subdirs = sorted(subdirs)
    shotnumbers = []
    for subdir in subdirs:
        subdir:Path = subdir
        if subdir.name.startswith("00"):
            intfile = get_background_int(subdir)
            background = readintfile(intfile)
            coefftype = get_coeff_type(intfile.with_suffix(".mcd"))
        else:
            try:
                intfile = list(subdir.glob("*.int"))[0]
            except IndexError:
                continue
            shotnumbers.append(int(subdir.name))
            times.append(pd.to_datetime(intfile.stem, 
                         format="%Y_%m_%d_%Hh_%Mmin_%Ss_%fus"))
            wavefront = readintfile(intfile)
            wavefront = wavefront-np.nanmean(wavefront) # remove piston
            wavefronts.append(wavefront)
            row = get_coeffs(intfile.with_suffix(".mcd"))
            coeffs.append(row)
    times = np.array(times)
    wavefronts = np.array(wavefronts)
    coeffs = np.array(coeffs)
    return times, wavefronts, coeffs, coefftype, shotnumbers, background

def get_background_int(parent):
    for path in Path(parent).glob("*"):
        if path.is_dir() and ("10" in path.name) and ("av" in path.name):
            intpath = list(path.glob("*.int"))[0]
    print("background path =", intpath)
    return intpath

def get_coeff_type(mcdfile):
    tree = ET.parse(mcdfile)
    root = tree.getroot()
    return root.find("./type").text

def collapse_path(intdir:str):
    name = intdir
    if re.search(r"^[a-zA-Z]:[\\/]", intdir):
        name = name[3:]
    name = name.replace("\\", r"%")
    name = name.replace("/", r"%")
    return name

def readintfile(intfile):
    header = pd.read_csv(intfile, header=None, skiprows=1,
                         nrows=1, sep='\\s+').loc[0].to_list()
    NDA = header[9]
    wavefront = pd.read_csv(intfile, header=None, skiprows=2,
                            sep='\\s+', na_values=NDA).to_numpy()
    if wavefront.shape[1] != header[1] or wavefront.shape[0] != header[2]:
        print("Array dimensions don't match with header")
    WVL = header[7]
    SSZ = header[5]
    wavefront = wavefront*WVL/SSZ
    return wavefront
    
def display_background(background, title, outdir):
    fig = plt.figure(layout="constrained", figsize=(12, 5))
    fig.suptitle(title)
    [ax0, ax1] = fig.subplots(1, 2)
    im0 = ax0.imshow(background, cmap="viridis")
    im_ratio = background.shape[0] / background.shape[1]
    fig.colorbar(im0, ax=ax0, fraction=0.047*im_ratio,
                 label="Wavefront $(\\mu m)$")
    ax0.set_title("With tilt")
    im1 = ax1.imshow(remove_tilt(background), cmap="viridis")
    fig.colorbar(im1, ax=ax1, fraction=0.047*im_ratio,
                 label="Wavefront $(\\mu m)$")
    ax1.set_title("No tilt")
    fig.savefig(outdir / "background.png")

def plotframes(wavefronts, times, minval, maxval, outdir:Path, figtitle,
               coeffs, RMS, removetilt, coefftype, terms, shotnumbers):
    legendre_names = (["L01 tilt 0", "L02 tilt 90", "L03 cylinder 0",
                       "L04 astig 45", "L05 cylinder 90", "L06 linear coma 0",
                       "L07", "L08", "L09 linear coma 90"]
                      + ["L" + i for i in map(str, range(10, 33))])
    zernike_names = (["Z01 tilt 0", "Z02 tilt 90", "Z03 focus", "Z04 astig 0",
                      "Z05 astig 45", "Z06 coma 0", "Z07 coma 90",
                      "Z08 spherical", "Z09 tref 0", "Z10 tref 90"]
                     + ["Z" + i for i in map(str, range(11, 33))])
    if coefftype == "legendre":
        labels = legendre_names
    elif coefftype == "zernike":
        labels = zernike_names
    labels = np.array(labels)[terms]
    coeffs = coeffs[:, terms]
    fig = plt.figure(layout="constrained", figsize=(10, 10))
    ax = fig.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]})
    ax0:plt.Axes = ax[0]
    ax1:plt.Axes = ax[1]
    ax2:plt.Axes = ax[2]
    ax1.plot(shotnumbers, RMS)
    ax1.set_xlim(shotnumbers[0], shotnumbers[-1])
    pad=0.05*(np.max(RMS)-np.min(RMS))
    miny = np.min(RMS)-pad
    maxy = np.max(RMS)+pad
    ax1.set_ylim(miny, maxy)
    [ax1line] = ax1.plot([shotnumbers[0], shotnumbers[0]], [miny, maxy],
                         color="red")
    ax1.set_ylabel("RMS $(\\mu m)$")

    ax2.plot(shotnumbers, coeffs, label=labels)
    pad=0.05*(np.max(coeffs)-np.min(coeffs))
    miny = np.min(coeffs)-pad
    maxy = np.max(coeffs)+pad
    ax2.set_ylim(miny, maxy)
    [ax2line] = ax2.plot([shotnumbers[0], shotnumbers[0]], [miny, maxy],
                         color="red")
    ax2.legend(loc="upper right")
    ax2.sharex(ax1)
    ax2.set_ylabel(f"{coefftype} standard coeff $(\\mu m)$")
    ax2.set_xlabel("Shot number")

    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title(figtitle)
    img = ax0.imshow(wavefronts[0], cmap="viridis", vmin=minval, vmax=maxval)
    if minval == None or maxval == None:
        cbar = fig.colorbar(img, label="Wavefront $(\\mu m)$",
                            format=lambda x, _: f"{x:.2f}")
    else:
        cbar = fig.colorbar(img, label="Wavefront $(\\mu m)$")
    txt1 = ax0.text(0.01, 0.01, "text1", ha="left", va="bottom",
                   transform=ax0.transAxes)
    txt2 = ax0.text(0.99, 0.97, "text2", transform=ax0.transAxes, ha="right",
                    va="top")
    txt3 = ax0.text(0.01, 0.99, "text2", transform=ax0.transAxes, ha="left",
                    va="top", fontsize=28)
    if removetilt:
        ax0.text(0.99, 0.94, "filtered tip/tilt", transform=ax0.transAxes,
                 ha="right", va="top")
    ax0.text(0.99, 0.89, "Background subtracted\nbefore shot 1",
             transform=ax0.transAxes, ha="right", va="top")
    for i in tqdm(range(0, len(shotnumbers))):
        fmttime = times[i].strftime("%Y_%m_%d_%Hh_%Mmin_%Ss_%fus")
        img.set_data(wavefronts[i])
        if minval == None or maxval == None:
            img.set_clim(np.nanmin(wavefronts[i]), np.nanmax(wavefronts[i]))
        RMS = np.sqrt(np.nanmean(wavefronts[i]**2))
        PV = (np.nanmax(wavefronts[i])-np.nanmin(wavefronts[i]))
        text = f"PV = {PV:.3f} $\\mu m$\nRMS = {RMS:.3f} $\\mu m$"
        txt1.set_text(text)
        txt2.set_text(times[i].strftime("%H:%M:%S.")
                      +str(times[i].microsecond//100000))
        txt3.set_text(shotnumbers[i])
        ax1line.set_xdata([shotnumbers[i], shotnumbers[i]])
        ax2line.set_xdata([shotnumbers[i], shotnumbers[i]])
        outfilename = outdir / "img" / Path(fmttime).with_suffix(".png")
        fig.savefig(outfilename)

def format_timer(tsec):
    minutes = int(np.abs(tsec) // 60)
    seconds = np.abs(tsec) % 60
    formatted = f"{minutes}:{seconds:04.1f}"
    if tsec < 0: 
        formatted = "-" + formatted
    else:
        formatted = "+" + formatted
    return formatted

def remove_tilt(wavefront):
    z = wavefront.reshape(np.prod(wavefront.shape))
    [ydim, xdim] = wavefront.shape
    xaxis = np.arange(xdim)
    yaxis = np.arange(ydim)
    X, Y = np.meshgrid(xaxis, yaxis)
    xcoords = X.reshape(np.prod(X.shape))
    ycoords = Y.reshape(np.prod(Y.shape))
    df = pd.DataFrame({"x": xcoords, "y": ycoords, "z":z})
    df = df.dropna()
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()
    popt, pcov = curve_fit(func, [x, y], z)
    z = z-func([x, y], *popt)
    z = z-np.mean(z)
    out = np.full(wavefront.shape, np.nan)
    for index, zvalue in enumerate(z):
        out[y[index], x[index]] = zvalue
    return out

def func(xy, a, b, c):
    return xy[0]*a + xy[1]*b + c

def pngwavefront(wavefront, filename):
    im = Image.fromarray(wavefront, "I")
    im.save(filename, compress_level=1)

def get_coeffs(mcdfile):
    tree = ET.parse(mcdfile)
    root = tree.getroot()
    values = root.find("./coefficients/data/values").text
    values = values.strip().split(" ")
    values = list(map(float, values))
    return np.array(values)

def get_cbar_scale(wavefronts, cbar_mode, cbar_values):
    if cbar_values == None: 
        if cbar_mode == "percentiles":
            cbar_values = (0.1, 99.9)
        elif cbar_mode == "manual":
            raise ValueError("cbar_values must be specified for "
                             "cbar_mode='manual'")
    truemin = np.nanmin(wavefronts)
    truemax = np.nanmax(wavefronts)
    print(f"True extents = [{truemin:.2f}, {truemax:.2f}]")
    if cbar_mode == "variable":
        lower = None
        upper = None
    elif cbar_mode == "minmax":
        lower = truemin
        upper = truemax
        print(f"Colorbar extents = [{lower:.2f}, {upper:.2f}]")
    elif cbar_mode == "manual":
        lower = cbar_values[0]
        upper = cbar_values[1]
        print(f"Colorbar extents = [{lower:.2f}, {upper:.2f}]")
    elif cbar_mode == "percentiles":
        lower = np.nanpercentile(wavefronts, cbar_values[0])
        upper = np.nanpercentile(wavefronts, cbar_values[1])
        print(f"Colorbar extents = [{lower:.2f}, {upper:.2f}]")
    else:
        raise ValueError(f"Unknown cbar_mode {cbar_mode}. "
                         "Must be 'variable', 'minmax', "
                         "'manual' or 'percentiles'")
    return lower, upper

if __name__ == "__main__":
    main()


# Colormap options:
# - gray
# - Greys - inverted grayscale
# - plasma
# - viridis
# - RdBu_r

