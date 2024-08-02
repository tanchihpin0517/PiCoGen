# PiCoGen (v1)

## Setup
### Clone
```sh
git clone --branch v1 https://github.com/tanchihpin0517/PiCoGen.git # or checkout branch `v1` after cloning
```

### SheetSage
Clone [SheetSage](https://github.com/chrisdonahue/sheetsage) into the root directory of this project.
```sh
git clone https://github.com/chrisdonahue/sheetsage.git
```
The directory structure should look like this:
```
(Project root)
.
|── ...
├── picogen
├── sheetsage
├── ...
```
We suggest to try SheetSage first to make sure the environment is set up correctly.
```sh
# Install SheetSage's dependencies
conda create -n sheetsage python=3.8 -y
conda install -y -n sheetsage Cython numpy
conda run -n sheetsage --no-capture-output pip install -r ./assets/requirements_sheetsage.txt
# Download SheetSage's model
conda run -n sheetsage --no-capture-output ./assets/sheetsage_prepare.sh
```
Note: SheetSage's environment is very complicated. We do not provide any support for SheetSage. If you have any problems, please refer to its repository. Besides, we do not use docker to run SheetSage since it fails on our machine (aws ec2 g5 type).

### PiCoGen
```sh
conda create -n picogen python=3.10 -y
conda run -n picogen --no-capture-output pip install -r ./assets/requirements_picogen.txt
```

### Download Pretrained Model
Download the pretrained model from [here](https://zenodo.org/records/11649613/files/model_00075000?download=1).
```sh
mkdir ./ckpt
wget https://zenodo.org/records/11649613/files/model_00075000?download=1 -O ./ckpt/model_00075000
```

## Run
Generate a piano cover from a youtube video or an audio file with the following command:
```sh
./picogen.sh \
    --input_url_or_file "[youtube url or input audio file]" \
    --output_dir "[output directory]" \
    --ckpt_file "[pretrained model]" \
    --vocab_file "[vocabulary file]" \
    --config_file "[model configuration]"
```
For example:
```sh
./picogen.sh \
    --input_url_or_file "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    --ckpt_file ./ckpt/model_00075000 \
    --output_dir ./output/never_gonna_give_you_up \
    --vocab_file ./assets/vocab.json \
    --config_file ./config/default.json
```


## Train

### Download Dataset
```sh
mkdir ./data
wget https://zenodo.org/records/11649613/files/pop1k7.zip\?download\=1 -O ./data/pop1k7.zip
unzip -d ./data ./data/pop1k7.zip
```

### Generate necessary files
```sh
./prepare.sh
```

### Train
```sh
./train.sh --batch_size 4
```
