import os
import json

from Utils.io import loadSingleMrc

from argparse import ArgumentParser

def getMRCData(file_path):
    volumeData, _ = loadSingleMrc(file_path, normalized=False, cache=False)

    data = {
        'file' : '',
        'size' : {
            'x' : volumeData.shape[2],
            'y' : volumeData.shape[1],
            'z' : volumeData.shape[0]
        },
        'ratio' : {
            'x' : 1.0,
            'y' : 1.0,
            'z' : 1.0
        },
        'bytesPerVoxel': 1,
        'usedBits': 8,
        'skipBytes': 0,
        'isLittleEndian': True,
        'isSigned': False,
        'addValue': 0
    }
    return data

def getJSONData(file_path):
    jsonFile = open(file_path)
    jsonData = json.load(jsonFile)

    return jsonData

if __name__=='__main__':
    parser = ArgumentParser('Duplicates JSON description file for new RAW or MRC volume file.')
    parser.add_argument('input_file_path', type=str, help='Input JSON file')
    parser.add_argument('target_file_path', type=str, help='Target RAW file')
    args = parser.parse_args()

    input_path = args.input_file_path
    target_path = args.target_file_path

    input_file = os.path.basename(input_path)
    target_file = os.path.basename(target_path)
    target_dir = os.path.dirname(target_path)

    data = []
    if input_file.lower().endswith(".json"):
        data = getJSONData(input_path)
    elif input_file.lower().endswith(".mrc"):
        data = getMRCData(input_path)
    
    data["file"] = target_file

    transferFunctionName = ''
    if 'Background' in target_file:
        transferFunctionName = 'tf-Background.json'
    elif 'Spikes' in target_file:
        transferFunctionName = 'tf-Spikes.json'
    elif 'Membrane' in target_file:
        transferFunctionName = 'tf-Membrane.json'
    elif 'Inner' in target_file:
        transferFunctionName = 'tf-Inner.json'
    else:
        transferFunctionName = 'tf-raw.json'
    data['transferFunction'] = transferFunctionName

    target_JSON_file = target_dir + '/' + target_file[0:target_file.rindex('.')] + ".json"

    with open(target_JSON_file, 'w') as f:
        json.dump(data, f, indent=4)