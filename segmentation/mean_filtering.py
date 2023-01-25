import numpy as np
import json
import scipy
import scipy.ndimage

from pathlib import Path
from argparse import ArgumentParser


def meanFilter(input_file, output_file, filter_size):
    # Read input file
    jsonFile = open(input_file)
    jsonData = json.load(jsonFile)

    # read raw file
    rawFile = open(str(Path(input_file).parent) + '/' + jsonData["file"])
    # transform to numpy array
    raw = np.fromfile(rawFile, dtype=np.uint8)
    # reshape to 3D array
    raw = raw.reshape((jsonData["size"]["z"], jsonData["size"]["y"], jsonData["size"]["x"]))

    # mean 3 filter kernel
    kernel = np.ones((filter_size, filter_size, filter_size), dtype=np.uint8) / (filter_size ** 3)

    # apply filter
    filtered = scipy.ndimage.convolve(raw, kernel, mode='nearest')

    # invert values
    filtered = 255 - filtered

    # save filtered volume to raw file
    filtered.tofile(output_file, format='uint8')

    # create json header file
    jsonData["file"] = Path(output_file).name
    with open(str(Path(output_file).parent) + '/' + Path(output_file).name[:-3] + 'json', "w") as jsonOut:
        jsonOut.write(json.dumps(jsonData, indent=4))
    


if __name__=='__main__':
    parser = ArgumentParser('Mean filter and invert 8-bit volume for visualization purposes')
    parser.add_argument('input_file', type=str, help='Input volume filename')
    parser.add_argument('output_file', type=str, help='Output volume filename')
    parser.add_argument('filter_size', type=int, default=3, nargs='?', help='Filter size')
    args = parser.parse_args()

    meanFilter(args.input_file, args.output_file, args.filter_size)