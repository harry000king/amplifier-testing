"""Harry King 08/05/2025

make_png_series() reads .int and .mcd files in a specified folder and plots a
series of wavefront maps. Each plot is saved as a .png image. On first run, the
program makes a local cache of the wavefront data so that loading is much
faster for subsequent runs.

video_from_pngs() can be used to combine the .png images into a video.

I recommend Nomacs image viewer for viewing .png files generated
https://nomacs.org/docs/getting-started/installation/. This allows easy
scrolling between images
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
from multiprocessing import Pool
from datetime import datetime
import cv2

def main():
    make_png_series(
        intdir = r"S:\Amplifier testing 2024\20250625\Series 1\HASO data",
        fileinterval = 100,
        starttime = "2025-06-25 10:40:00", # "2025-06-25 11:44:00"
        stoptime = "2025-06-25 11:00:00", # "2025-06-25 11:48:00"
        removetilt = False,
        cbar_mode = "percentiles",
        background_time = ["2025-06-25 10:45:40", "2025-06-25 10:45:41"] # ["2025-06-25 10:45:40", "2025-06-25 10:47:30"]
    )
    # video_from_pngs(r"C:\Users\harry\Documents\py_vid\out\20250711-101355")

def make_png_series(intdir:str, fileinterval:int=1, starttime:str=None,
              stoptime:str=None, removetilt:bool=True, terms=None,
              cbar_mode:str="percentiles", cbar_values:tuple=None,
              background_time:str|tuple=None):
    """Run this function first to generate series of .png files.
    
    Args:
        * intdir: Folder path containing .int and .mcd files
        * fileinterval: Number of files to skip between each plotted wavefront.
          Increase this number to plot faster at lower time resolution.
        * starttime: Time to begin series. String with format "2025-03-10
          09:17:36" or None.
        * stoptime: Time to end series.
        * removetilt: Switch for removing tip/tilt from wavefront maps.
        * terms: Indices of zernike or legendre coefficients to plot in graph.
          It is recommended to use the default unless fine control is needed.
          np.arange(0, 10) for Zernikes while reacting to removetilt flag.
        * cbar_mode: Determines how colorbar will be scaled - 4 options:
            - "variable" - autoscale for every frame
            - "minmax" - fixed scale based on global min/max wavefront
            - "manual" - cbar_values must be manually specified
            - "percentiles" - cbar_values defaults to (0.1, 99.9) but can
              also be manually specified
        * cbar_values: Behaviour depends on cbar_mode (see above)
        * background_time: Timestamp of frame to use as background for zeroing
          wavefront. Behaviour depends on type:
            - None: no background subtraction
            - "2025-06-25 11:44:00": round this to nearest frame
            - ("2025-06-25 11:44:00", "2025-06-25 11:48:00"): average over
              this time range
    """
    parent = Path.home() / "Documents" / "py_vid"
    safe_mkdir(parent)
    safe_mkdir(parent / "cache")
    safe_mkdir(parent / "out")
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = parent / "out" / now
    outdir.mkdir()
    (outdir / "img").mkdir()
    cachedir = parent / "cache" / collapse_path(intdir)
    if cachedir.exists():
        times, wavefronts, coeffs, coefftype = get_cache(cachedir)
    else:
        temp = get_direct(intdir)
        times, wavefronts, coeffs, coefftype = temp
        save_cache(cachedir, times, wavefronts, coeffs, coefftype)
    if terms == None:
        terms = get_default_terms(removetilt, terms, coefftype)
    background = get_background(times, wavefronts, background_time)
    times, wavefronts, coeffs = downselect(times, wavefronts, coeffs, 
                                           fileinterval, starttime, stoptime)
    message = ""
    if background is not None:
        message += "Subtracting background, "
    if removetilt:
        message += "filtering tip/tilt, "
    message += "Computing RMS"
    print(message)
    RMS = []
    for i, wavefront in tqdm(enumerate(wavefronts)):
        if background is not None:
            pass
        if removetilt:
            wavefronts[i] = remove_tilt(wavefront)
        RMS.append(np.sqrt(np.nanmean(wavefront**2)))
    RMS = np.array(RMS)
    minval, maxval = get_cbar_scale(wavefronts, cbar_mode, cbar_values)
    cores = os.cpu_count()
    index = np.arange(0, len(times), 1)
    index_split = np.array_split(index, cores)
    args = []
    for i in range(0, cores):
        args.append([wavefronts[index_split[i]], times, minval, maxval, outdir,
                     intdir, coeffs, RMS, index_split[i], removetilt,
                     coefftype, terms])
    print(f"Plotting with {cores} subprocesses")
    with Pool(cores) as p:
        p.starmap(plotframes, args)
    os.startfile(outdir)

def video_from_pngs(outdir, fps=20):
    """Run this function after make_png_series()"""
    print("Building .mp4 from .png series")
    outdir = Path(outdir)
    pngdir = outdir / "img"
    outfile = outdir / "out.mp4"
    if outfile.exists():
        print("Video already exists in outdir => overwriting")
    files = sorted(pngdir.glob("*.png"))
    firstframe = cv2.imread(files[0])
    height, width, _ = firstframe.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
    for file in tqdm(files):
        frame = cv2.imread(file)
        video.write(frame)
    video.release()
    print("Saved ->", outfile)

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

def get_cache(cachedir):
    print(f"Loading cache from {cachedir}  ...  ", end="", flush=True)
    times = np.load(cachedir / "times.npy", allow_pickle=True)
    wavefronts = np.load(cachedir / "wavefronts.npy", allow_pickle=True)
    coeffs = np.load(cachedir / "coeffs.npy", allow_pickle=True)
    coefftype = str(np.load(cachedir / "coefftype.npy", allow_pickle=True))
    print("Done")
    return times, wavefronts, coeffs, coefftype

def save_cache(cachedir, times, wavefronts, coeffs, coefftype):
    cachedir.mkdir()
    print(f"Saving cache -> {cachedir}")
    np.save(cachedir / "times.npy", times)
    np.save(cachedir / "wavefronts.npy", wavefronts)
    np.save(cachedir / "coeffs.npy", coeffs)
    np.save(cachedir / "coefftype.npy", coefftype)

def get_direct(intdir):
    print(f"Loading .int, .mcd files from {intdir}")
    if not(Path(intdir).exists()):
        raise FileNotFoundError("Cannot find source folder. "
                                "Perhaps wake up network drive?")
    times = []
    wavefronts = []
    coeffs = []
    firstfile = True
    for file in tqdm(sorted(Path(intdir).glob("*.int"))):
        datetime = pd.to_datetime(file.stem, 
                                  format="%Y_%m_%d_%Hh_%Mmin_%Ss_%fus")
        wavefront = readintfile(file)
        wavefront = wavefront-np.nanmean(wavefront) # remove piston
        wavefronts.append(wavefront)
        times.append(datetime)
        row = get_coeffs(file.with_suffix(".mcd"))
        coeffs.append(row)
        if firstfile:
            coefftype = get_coeff_type(file.with_suffix(".mcd"))
            firstfile = False
    times = np.array(times)
    wavefronts = np.array(wavefronts)
    coeffs = np.array(coeffs)
    return times, wavefronts, coeffs, coefftype

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

def downselect(times, wavefronts, coeffs, fileinterval, starttime,
               stoptime):
    sliceidx = pd.Series(np.arange(len(times)), index=times)
    sliceidx = sliceidx[starttime:stoptime].to_numpy()
    sliceidx = sliceidx[::fileinterval]
    return (times[sliceidx], wavefronts[sliceidx], coeffs[sliceidx])

def get_background(times, wavefronts, background_time):
    if background_time == None:
        return None
    elif type(background_time) == str:
        sliceidx = pd.Series(np.arange(len(times)), index=times)
        idx = sliceidx.index.get_indexer([background_time], method="nearest")[0]
        print("Using background at:", times[idx])
        return wavefronts[idx]
    elif len(background_time) == 2:
        sliceidx = pd.Series(np.arange(len(times)), index=times)
        temp = sliceidx[background_time[0]:background_time[1]]
        firsttime = temp.index[0]
        lasttime = temp.index[-1]
        myslice = temp.to_numpy()
        print(f"Generating background: averaging {len(myslice)} frames"
              f" {firsttime} -> {lasttime}")
        background = np.nanmean(wavefronts[myslice], axis=0)
        display_background(background, firsttime, lasttime, len(myslice))
        return background
    else:
        raise ValueError("background_time must be None, a string or a tuple "
                         "of two strings")
    
def display_background(background, firsttime, lasttime, frames):
    fig = plt.figure(layout="constrained", figsize=(12, 5))
    fig.suptitle(f"Averaged background: {firsttime} -> {lasttime} "
                  f"({frames} frames)\nWill be subtracted from every frame")
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
    fig.savefig("background.png")

def plotframes(wavefronts, times, minval, maxval, outdir:Path, figtitle,
               coeffs, RMS, index, removetilt, coefftype, terms):
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
    ax1.plot(times, RMS)
    ax1.set_xlim(times[0], times[-1])
    pad=0.05*(np.max(RMS)-np.min(RMS))
    miny = np.min(RMS)-pad
    maxy = np.max(RMS)+pad
    ax1.set_ylim(miny, maxy)
    [ax1line] = ax1.plot([times[0], times[0]], [miny, maxy], color="red")
    ax1.set_ylabel("RMS $(\\mu m)$")

    ax2.plot(times, coeffs, label=labels)
    pad=0.05*(np.max(coeffs)-np.min(coeffs))
    miny = np.min(coeffs)-pad
    maxy = np.max(coeffs)+pad
    ax2.set_ylim(miny, maxy)
    [ax2line] = ax2.plot([times[0], times[0]], [miny, maxy], color="red")
    ax2.legend(loc="upper right")
    ax2.sharex(ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    xformatter = mdates.DateFormatter('%H:%M:%S')
    ax2.xaxis.set_major_formatter(xformatter)
    ax2.set_ylabel(f"{coefftype} standard coeff $(\\mu m)$")

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
    txt2 = ax0.text(0.97, 0.97, "text2", transform=ax0.transAxes, ha="right",
            va="top")
    if removetilt:
        ax0.text(0.97, 0.94, "filtered tip/tilt", transform=ax0.transAxes,
                 ha="right", va="top")
    for i in tqdm(index):
        fmttime = times[i].strftime("%Y_%m_%d_%Hh_%Mmin_%Ss_%fus")
        wavefront = wavefronts[i-index[0]]
        img.set_data(wavefront)
        if minval == None or maxval == None:
            img.set_clim(np.nanmin(wavefront), np.nanmax(wavefront))
        RMS = np.sqrt(np.nanmean(wavefront**2))
        PV = (np.nanmax(wavefront)-np.nanmin(wavefront))
        text = f"PV = {PV:.3f} $\\mu m$\nRMS = {RMS:.3f} $\\mu m$"
        txt1.set_text(text)
        txt2.set_text(times[i].strftime("%H:%M:%S.")
                      +str(times[i].microsecond//100000))
        ax1line.set_xdata([times[i], times[i]])
        ax2line.set_xdata([times[i], times[i]])
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

# Runtime warning: Mean of empty slice
# Also display background for str option
# Background_time slice limits don't quite work in the way I was expecting - they aren't upper and lower bounds
# Add nice picture to readme.md for github