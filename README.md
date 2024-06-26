# PiCoGen

## Setup
### SheetSage
Clone [SheetSage](https://github.com/chrisdonahue/sheetsage) into the root directory of this project.
```
git clone https://github.com/chrisdonahue/sheetsage.git

# Project root
# .
# ├── ...
# ├── picogen
# ├── sheetsage
# ├── ...
```
We suggest to try SheetSage first to make sure the environment is set up correctly.
```sh
# Install SheetSage's dependencies
conda create -n sheetsage python=3.8 -y
conda install -y -n sheetsage Cython numpy
conda run -n sheetsage --no-capture-output pip install -r ./asset/requirements_sheetsage.txt
# Download SheetSage's model
conda run -n sheetsage --no-capture-output ./asset/sheetsage_prepare.sh
```
Note: SheetSage's environment is very complicated. We do not provide any support for SheetSage. If you have any problems, please refer to its repository. Besides, we do not use docker to run SheetSage since it fails on our machine (aws ec2 g5 type).

### PiCoGen
```sh
conda create -n picogen python=3.10 -y
conda run -n picogen --no-capture-output pip install -r ./asset/requirements_picogen.txt
```

### Download Pretrained Model
Download the pretrained model from [here](https://zenodo.org/records/11649613/files/model_00075000?download=1).

## Run
Generate a piano cover from a youtube video or an audio file with the following command:
```sh
./picogen.sh \
    --input_url_or_file "[youtube url or input audio file]" \
    --output_dir "[output directory]" \
    --ckpt_file "[pretrained model]"
```
If you have extracted leadsheets with SheetSage, you can specify the leadsheet directory (named with an UUID by default):
```sh
./picogen.sh \
    --leadsheet_dir "[SheetSage output directory]" \
    --output_dir "[output directory]" \
    --ckpt_file "[pretrained model]"
```


## Train PiCoGen

### Download Dataset
```sh
git clone https://github.com/tanchihpin0517/dataset-pop1k7.git data/pop1k7
```

### Train
```sh
./train.sh --batch_size 4
```
