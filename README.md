# picogen2

## Quickstart: generate your piano cover!
PiCoGen2 is an automatic piano cover generation system which utilize [SheetSage](https://github.com/chrisdonahue/sheetsage) to extract music feature from input audio and produce its cover version of the orignal piece.
You can run this model with [Docker](https://docs.docker.com/) or local environments. Ensure you are running Linux with GPU memory larger than 16GB.

### Requirements
* Ubuntu OS (we haven't tested on other systems)
* GPU with more than 16GB memory
* `ffmpeg` installed

### Quickstart
You can reproduce the [demo]() just with these commands:
```sh
mkdir ./demo_output/never_gonna_give_you_up
./demo.sh --docker_image "tanchihpin0517/picogen2:latest-full"
```
That will download the [song](https://www.youtube.com/watch?v=dQw4w9WgXcQ) from Youtube and generate its piano cover saved in `./demo_output/never_gonna_give_you_up`.



### Run with Docker
Make sure [Docker installed](https://docs.docker.com/desktop/install/linux-install/) on your system. You can then use this command to run PiCoGen2 in a Docker container:
```sh
./infer \
    --input_url "youtube link" \ # or --input_audio "path of audio file"
    --output_dir "where to save the outputs" \
    --docker_image "image_name:image_tag"
```
PiCoGen2's Docker image has been published to [Dockerhub](https://hub.docker.com/repository/docker/tanchihpin0517/picogen2/general). The repository is named with `tanchihpin0517/picogen2`. There are two tags that can be found in the repository: `latest` and `latest-full`.

- The image tagged with `latest` contains the complete Python environment to run PiCoGen2 that are installed by Conda.
- Another one tagged with `latest-full` includes not only all necessary Python packages but **pre-downloaded model checkpoints** of beat trackers, SheetSage, and PiCoGen's decoder.


### Run in local
We recommand to use [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to manage the Python environment.

> `mpi4py` is required by SheetSage, however, not included in the dependencies of PiCoGen2. I recommend to install it with `conda install mpi4py`.

If you use Conda as Python environment manager, you can follow the instruction to build your runtime environment:
```sh
# the content of ./setup_env.sh
conda create -n picogen2 python=3.10 -y
conda install -n picogen2 mpi4py Cython -y
conda run -n picogen2 --no-capture-output pip install .
```
After these are settled down, you can also use the same script `infer.sh` to run our model in the local environment.
```sh
./infer \
    --input_url "youtube link" \ # or --input_audio [path of audio file] 
    --output_dir "where to save the outputs"
```



## Python interface
We provide Python APIs for users who would like to run PiCoGen2 in their own Python codes like this:
```python
import tempfile
from pathlib import Path

from mirtoolkit import beat_this, sheetsage

import picogen2 # make sure picogen2 is installed
import picogen2.assets # remove this if you don't use the default testing song

def main():
    audio_file = picogen2.assets.test_song() # input file
    output_dir = tempfile.TemporaryDirectory() # output directory

    # initialize
    tokenizer = picogen2.Tokenizer()
    model = picogen2.PiCoGenDecoder.from_pretrained(device="cuda")

    # detect beats
    beats, downbeats = beat_this.detect(audio_file)
    beat_information = {"beats": beats.tolist(), "downbeats": downbeats.tolist()}

    # extract feature
    sheetsage_output = sheetsage.infer(audio_path=audio_file, beat_information=beat_information)

    # generate piano cover
    out_events = picogen2.decode(
        model=model,
        tokenizer=tokenizer,
        beat_information=beat_information,
        melody_last_embs=sheetsage_output["melody_last_hidden_state"],
        harmony_last_embs=sheetsage_output["harmony_last_hidden_state"],
    )

    # save results
    (Path(output_dir.name) / "piano.txt").write_text("\n".join(map(str, out_events)))
    tokenizer.events_to_midi(out_events).dump(Path(output_dir.name) / "piano.mid")


if __name__ == "__main__":
    main()
```

## Q & A
> Can I run PiCoGen2 without GPU?
- Unfortunately, we still haven't implemented CPU version of PiCoGen2. Currently we don't have a plan for that but we welcome any contribution.
