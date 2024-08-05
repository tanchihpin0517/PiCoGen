import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(0)

IGNORE_MISSING_TAGS = False
MAX_SONG_DURATION = 30

SONGS_DIR = Path("./static/songs")
OUTPUT_FILE_EXAMPLES = "./examples.html"
TEMPLATE_FILE_EXAMPLES = "./generation/templates/examples.html"
TEMPLATE_FILE_EXAMPLES_NAV_ITEM = "./generation/templates/examples_nav_item.html"
TEMPLATE_FILE_EXAMPLES_SONG = "./generation/templates/examples_song.html"


class Template:
    def __init__(self, path):
        self.path = Path(path)
        self.text = self.path.read_text()
        self.editions = defaultdict(list)

    def write(self, tag, content):
        if not IGNORE_MISSING_TAGS and f"%{tag}%" not in self.text:
            raise ValueError(f"Tag '{tag}' does not exist in the target string.")
        if isinstance(content, Template):
            content = content.render()
        self.editions[tag].append(content)

    def render(self):
        output = self.text
        for tag, content in self.editions.items():
            output = output.replace(f"%{tag}%", "".join(content))
        return output

    def __str__(self):
        return self.render()


def main():
    generate_picogen2_experiment()


def generate_picogen2_experiment():
    # get file list
    songs_dir = Path(SONGS_DIR)
    songs = []
    for genre_dir in sorted(list(songs_dir.iterdir())):
        if genre_dir.is_file():
            continue
        genre = genre_dir.name
        for song_dir in sorted(list(genre_dir.iterdir())):
            if song_dir.is_file():
                continue
            song_id = song_dir.name
            metadata = json.loads((song_dir / "metadata.json").read_text())
            songs.append(
                {
                    "genre": genre,
                    "metadata": metadata,
                    "midi_files": {},
                    "audio_files": {},
                }
            )

            for model in ["human", "picogen2", "picogen1", "pop2piano", "audio2midi"]:
                midi_file = song_dir / f"{model}.mid"
                audio_file = song_dir / f"{model}_{MAX_SONG_DURATION}.ogg"
                assert (
                    midi_file.exists()
                ), f"Missing {model} file for `{genre} {song_id}`"
                assert (
                    audio_file.exists()
                ), f"Missing {model} file for `{genre} {song_id}`"
                songs[-1]["midi_files"][model] = str(midi_file)
                songs[-1]["audio_files"][model] = str(audio_file)

    output_html_file = Path(OUTPUT_FILE_EXAMPLES)

    content = Template(TEMPLATE_FILE_EXAMPLES)

    for i, song in enumerate(songs[:]):
        song_idx = str(i + 1).zfill(2)
        song_tag = f"song_nav_{song_idx}"

        nav_item = Template(TEMPLATE_FILE_EXAMPLES_NAV_ITEM)
        nav_item.write("song_tag", song_tag)
        nav_item.write("item", (song_idx + ". ") + song["metadata"]["pop_title"][:])

        content.write("nav_bar", nav_item)
        print(song["metadata"]["pop_title"])

        list_item = Template(TEMPLATE_FILE_EXAMPLES_SONG)
        list_item.write("title", (song_idx + ". ") + song["metadata"]["pop_title"])
        list_item.write("song_tag", song_tag)
        list_item.write(
            "song_link", f"https://www.youtube.com/watch?v={song['metadata']['pop_id']}"
        )
        list_item.write(
            "human_link",
            f"https://www.youtube.com/watch?v={song['metadata']['piano_id']}",
        )
        list_item.write("audio_human", song["audio_files"]["human"])
        list_item.write("audio_picogen2", song["audio_files"]["picogen2"])
        list_item.write("audio_picogen1", song["audio_files"]["picogen1"])
        list_item.write("audio_pop2piano", song["audio_files"]["pop2piano"])
        list_item.write("audio_audio2midi", song["audio_files"]["audio2midi"])
        if int(song_idx) % 2 == 0:
            list_item.write("bg_style", "bg_grey")

        content.write("song_list", list_item)

    output_html_file.write_text(str(content))


if __name__ == "__main__":
    main()
