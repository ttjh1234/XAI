import csv
import os
import datasets

_CITATION = """\
@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew and Potts, Christopher},
  booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},
  pages={1631--1642},
  year={2013}
}
"""

_DESCRIPTION = """\
The Stanford Sentiment Treebank consists of sentences from movie reviews and
human annotations of their sentiment. The task is to predict the sentiment of a
given sentence. We use the two-way (positive/negative) class split, and use only
sentence-level labels.
"""

_HOMEPAGE = "https://nlp.stanford.edu/sentiment/"

_LICENSE = "Unknown"

def split_generators(dl_manager):
    _URL="https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
    dl_dir = dl_manager.download_and_extract(_URL)
    return [
        datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "file_paths": dl_manager.iter_files(dl_dir),
                "data_filename": "train.tsv",
            },
        ),
        datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            gen_kwargs={
                "file_paths": dl_manager.iter_files(dl_dir),
                "data_filename": "dev.tsv",
            },
        ),
        datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={
                "file_paths": dl_manager.iter_files(dl_dir),
                "data_filename": "test.tsv",
            },
        ),
    ]

def generate_examples(file_paths, data_filename):
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if filename == data_filename:
            with open(file_path, encoding="utf8") as f:
                reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                for idx, row in enumerate(reader):
                    yield idx, {
                        "idx": row["index"] if "index" in row else idx,
                        "sentence": row["sentence"],
                        "label": int(row["label"]) if "label" in row else -1,
                    }