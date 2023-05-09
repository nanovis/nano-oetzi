
# Example segmentation call with the test data

Download project's test data to the ./data folder.

```bash
./tests/get_testdata.sh
```

Build Docker image

```bash
docker build . -t nanooetzi -f ./Dockerfile
```


Run segmentation with the four-classes model

```bash
docker run --rm \
  -v `pwd`/data/:/data\
  -v /data/tmp/r:/data/tmp/r\
   nanooetzi\
   python ./inference_script.py /data/ts_16_bin4-256x256.json\
      -m ../models/four_classes_model.ckpt\
      /data/tmp/r
```
