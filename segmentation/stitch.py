#### Stitching 4-class
from fileinput import filename
import os
import json
import torch
import numpy as np

import Utils.stitching_all_class as st

from pathlib import Path
from argparse import ArgumentParser


def stitchVolumes(prefix, exp_path, output_path, json_file, tile_locations_file, output_prefix):
    # json_file_path = exp_path + json_file
    # jsonFile = open(json_file_path)
    jsonFile = open(json_file)
    jsonData = json.load(jsonFile)

    overlap = jsonData['overlap']
    vol_locs = jsonData['split_coords']
    slices = vol_locs[0][0][1]+1
    vol_num = len(vol_locs)
    vol_numX = 0
    for i in range(vol_num):
        if vol_locs[i][1][0] == 0:
            vol_numX += 1
    vol_numY = int(vol_num / vol_numX)

    volX = vol_locs[-1][-1][-1]+1 # 3
    volY = vol_locs[-1][-2][-1]+1 # 4

    print('slices: ', slices, ' volX: ', volX, ' volY: ', volY)

    volume_background = torch.zeros(slices, volY, volX)
    volume_membrane = torch.zeros(slices, volY, volX)
    volume_spikes = torch.zeros(slices, volY, volX)
    volume_inner = torch.zeros(slices, volY, volX)

    volume_size_x = min(512, volX)
    volume_size_y = min(512, volY)

    mask_vols = []
    for i in range(vol_num):
        mask_vols.append(torch.ones(slices, volume_size_y, volume_size_x))

    x_thr = overlap[2]
    y_thr = overlap[1]

    # print('vol_numY: ' + str(vol_numY))
    # print('vol_numX: ' + str(vol_numX))

    for i in range(x_thr):
        for j in range(vol_num):
            if j % vol_numX == 0:
                mask_vols[j][:, :, -i] *= i / x_thr
            
            elif j % vol_numX == (vol_numX - 1):
                mask_vols[j][:, :, i] *= i / x_thr
            
            else:
                mask_vols[j][:, :, i] *= i / x_thr
                mask_vols[j][:, :, -i] *= i / x_thr
            
    for i in range(y_thr):
        for j in range(vol_num):
            if j < vol_numX:
                mask_vols[j][:, -i, :] *= i / y_thr
            
            elif j >= (vol_num - vol_numX):
                mask_vols[j][:, i, :] *= i / y_thr
            
            else:
                mask_vols[j][:, i, :] *= i / y_thr
                mask_vols[j][:, -i, :] *= i / y_thr

    for v_num in range(vol_num):
    # for vol_num in range(1,2):
        print(v_num)

        # [batch, batch_size, num_class, TILE]
        blocks = torch.load(exp_path + prefix + str(v_num) + '.pt')

        # [batch, batch_size, batch_size * batch, LOC]
        locations = torch.load(tile_locations_file + str(v_num) + '.pt')

        overlap_size = 32 - 1
        block_size = 128

        max_locations = [0, 0, 0]
        min_locations = [slices * 2, volY * 2, volX * 2]

        # Function calls:
        st.find_block_limits_2(locations, min_locations, max_locations)

        volume_0 = torch.zeros(slices, volume_size_y, volume_size_x)
        class_num = 0
        st.alpha_blended_stitching_all_class(blocks, volume_0, locations, min_locations, max_locations, block_size, overlap_size, class_num)

        volume_1 = torch.zeros(slices, volume_size_y, volume_size_x)
        class_num = 1
        st.alpha_blended_stitching_all_class(blocks, volume_1, locations, min_locations, max_locations, block_size, overlap_size, class_num)

        volume_2 = torch.zeros(slices, volume_size_y, volume_size_x)
        class_num = 2
        st.alpha_blended_stitching_all_class(blocks, volume_2, locations, min_locations, max_locations, block_size, overlap_size, class_num)

        volume_3 = torch.zeros(slices, volume_size_y, volume_size_x)
        class_num = 3
        st.alpha_blended_stitching_all_class(blocks, volume_3, locations, min_locations, max_locations, block_size, overlap_size, class_num)

        x0 = vol_locs[v_num][0][0]
        x1 = vol_locs[v_num][0][1]+1
        y0 = vol_locs[v_num][1][0]
        y1 = vol_locs[v_num][1][1]+1
        z0 = vol_locs[v_num][2][0]
        z1 = vol_locs[v_num][2][1]+1

        volume_background[x0:x1, y0:y1, z0:z1] += volume_0 * mask_vols[v_num]
        volume_membrane[x0:x1, y0:y1, z0:z1] += volume_1 * mask_vols[v_num]
        volume_spikes[x0:x1, y0:y1, z0:z1] += volume_2 * mask_vols[v_num]
        volume_inner[x0:x1, y0:y1, z0:z1] += volume_3 * mask_vols[v_num]

    # st.display_volume_slice_comparison_all_class(volume_background, volume_membrane, volume_spikes, volume_inner, 128)

    volume_background = torch.flip(volume_background, [1])
    volume_membrane = torch.flip(volume_membrane, [1])
    volume_spikes = torch.flip(volume_spikes, [1])
    volume_inner = torch.flip(volume_inner, [1])

    filename = output_path + output_prefix + '-Background.raw'
    st.save_to_binary_file(volume_background, filename)

    filename = output_path + output_prefix + '-Membrane.raw'
    st.save_to_binary_file(volume_membrane, filename)

    filename = output_path + output_prefix + '-Spikes.raw'
    st.save_to_binary_file(volume_spikes, filename)

    filename = output_path + output_prefix + '-Inner.raw'
    st.save_to_binary_file(volume_inner, filename)


if __name__=='__main__':
    parser = ArgumentParser('Stitch depth * 512 * 512 volumes into one volume')
    parser.add_argument('file_prefix', type=str, help='Volume filename prefix')
    parser.add_argument('volume_path', type=str, help='Path to volume')
    parser.add_argument('save_dir', type=str, help='Path to save torch volume dir')
    parser.add_argument('volume_chunks_file', type=str, help='JSON file containing volume chunk locations')
    parser.add_argument('volume_tiles_file', type=str, help='JSON file containing volume tile locations')
    parser.add_argument('output_prefix', type=str, help='Output volume prefix')
    args = parser.parse_args()
    
    save_dir = Path(args.save_dir)
    if not save_dir.exists(): os.mkdir(save_dir)

    # # Load original volume
    # if args.filename.endswith(".json") or args.filename.endswith(".JSON"):
    #     volumeData = loadJSONVolume(args.filename)
    # elif args.filename.endswith(".mrc"):
    #     volumeData, _ = loadSingleMrc(args.filename, normalized=True, cache=True)

    # if isinstance(volumeData, np.ndarray):
    #     volumeData = torch.from_numpy(volumeData)
    stitchVolumes(args.file_prefix, args.volume_path, args.save_dir, args.volume_chunks_file, args.volume_tiles_file, args.output_prefix)