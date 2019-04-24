# Sheet ID

This repository contains the code used in "Sheet Music Identification Using Measure-Based CNN Features" by Hubinger, Khant, Kurashige, Amin, and Tsai.

## Installation

To install this repository, you will need to

1. `git clone https://github.com/evhub/score-retrieval.git`,
2. download the data to `./data`,
3. install [the CNN code](https://github.com/evhub/cnnimageretrieval-pytorch),
4. install [the measure segmentation code](https://github.com/aditya-khant/sheet-id-splitter), and
5. `make install`.

## Usage

First, you will need to ensure your data is properly set up. If you only have PDFs but not images, you will need to run `make pdfs-to-images`.

Then, to run the system, simply
```
python ./score_retrieval/run_all.py --alg <alg_name>
```
where `<alg_name>` is one of the following:

- `measure_segmentation` to run our best system using measure segmentation,
- `vgg_measure_segmentation` to run base vgg-gem with measure segmetnation, and
- `tuned_measure_segmentation` to run measure segmentation utilizing a fine-tuned network.

To run `tuned_measure_segmentation`, you will need to have previously [run fine-tuning on the training data](https://github.com/evhub/cnnimageretrieval-pytorch).
