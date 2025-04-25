[![](https://img.shields.io/badge/pytorch%20-1.9.1%20-success?logo=pytorch)]()
[![](https://img.shields.io/badge/python%20-3.8%20-sucess?logo=python)]()
[![](https://img.shields.io/badge/tqdm%20-4.65.0%20-success?logo=tqdm)]()
[![](https://img.shields.io/badge/linux%20-gray?logo=linux)]()

# Spherical Embeddings for Atomic Relation Projection Reaching Complex Logical Query Answering

This is the implementation of the paper
[Spherical Embeddings for Atomic Relation Projection Reaching Complex Logical Query Answering](https://dl.acm.org/doi/10.1145/3696410.3714747) in WWW '25.

<!-- Chau Nguyen, [Tim French](https://scholar.google.com.au/citations?user=zcqX-AgAAAAJ&hl=en&oi=ao), [Michael Stewart](https://scholar.google.com.au/citations?user=8-kgpZkAAAAJ&hl=en&oi=ao), [Melinda Hodkiewicz](https://scholar.google.com.au/citations?hl=en&user=1JGboosAAAAJ) [Wei Liu](https://scholar.google.com.au/citations?user=o_u17HMAAAAJ&hl=en)-->

## Acknowledgements

We acknowledge the code of
[KGReasoning](https://github.com/snap-stanford/KGReasoning) and the code of
[QTO](https://github.com/bys0318/QTO) for their contributions.

## Getting started

### Step 1: Data preparation

- Download the datasets [here](http://snap.stanford.edu/betae/KG_data.zip), then move `KG_data.zip` to `./sphere/` directory

- Unzip `KG_data.zip` to `./sphere/data/`:

  ```bash
  cd sphere/
  unzip -d data KG_data.zip
  ```

### Step 2: Dependencies installation

- For `condaers`:
  ```bash
  conda env create -f requirements.yml
  conda activate sphere
  ```
- For `pipers`:

  ```bash
  python -m venv venv
  source ./venv/bin/activate
  pip install -r requirements.txt
  ```

### Step 3: Model

- There are three datasets: FB15k-237, FB15k, NELL995
- We recommend that you start with the FB15k-237 and FB15k first for faster implementation and less GPU memory requirements than that with the NELL dataset.

#### Step 3.1: Training query embeddings models

1. Run the scripts `scripts/*.sh` at the parent directory `sphere/` to train query embeddings models (GQE, Query2Box, SpherE) for the default dataset `FB15k-237`
2. Uncomment others in `*.sh` to train models using other datasets `(FB15k/NELL995)`. For example, type the following command to train a specific model:

- **SpherE** (trained using 1p queries only)

  ```bash
  scripts/sphere_1p.sh
  ```

- **GQE**

  ```bash
  scripts/gqe.sh
  ```

- **GQE** (trained using 1p queries only)

  ```bash
  scripts/gqe_1p.sh
  ```

- **Query2Box**

  ```bash
  scripts/query2box.sh
  ```

- **Query2Box** (trained using 1p queries only)

  ```bash
  scripts/query2box_1p.sh
  ```

#### Step 3.2: Move the checkpoint

- After successfully training a specific model, copy or move its checkpoint file `checkpoint`
  under the `logs/` directory to the directory `clqa/pre_trained_clqa/`, to prepare for the next step. For example,
  the final path of the checkpoint file is as follows:

  ```bash
  clqa/pre_trained_clqa/checkpoint
  ```

  or rename it

  ```bash
  clqa/pre_trained_clqa/FB15k-237_sphere_256
  ```

#### Step 3.3: Generalizing query embeddings to answer complex logical queries using fuzzy logic

1. Run the scripts `scripts/*.sh` at the parent directory `sphere/` to generate atomic query matrix (neural adjacency matrix) and to generalize query embeddings models (GQE, Query2Box, SpherE) using the default dataset `FB15k-237` for complex logical queries.
2. Ensure that this option `--clqa_path`, for example, `--clqa_path clqa/pre_trained_clqa/FB15k-237_sphere_256` matches to the name of checkpoint of pre-trained model, e.g. `FB15k-237_sphere_256`. You can rename `checkpoint` to `FB15k-237_sphere_256` for distinguishing other checkpoints.
3. Ensure that this option `-d`, for example `-d 256`, matches to the dimension option `-d 256` of pre-trained model.
4. Uncomment others in `*.sh` for other datasets `(FB15k/NELL995)`.

- **SpherE** (trained using 1p queries only)

  ```bash
  scripts/sphere_qto.sh
  ```

- **GQE**

  ```bash
  scripts/gqe_qto.sh
  ```

- **GQE** (1p version)

  ```bash
  scripts/gqe_qto_1p.sh
  ```

- **Query2Box**

  ```bash
  scripts/query2box_qto.sh
  ```

- **Query2Box** (1p version)

  ```bash
  scripts/query2box_qto_1p.sh
  ```

## Citation

If you find this code useful for your research, please consider citing the following paper:

```bib
@inproceedings{nguyen2025spherical,
    author = {Nguyen, Chau D. M. and French, Tim and Stewart, Michael and Hodkiewicz, Melinda and Liu, Wei},
    title = {Spherical Embeddings for Atomic Relation Projection Reaching Complex Logical Query Answering},
    year = {2025},
    isbn = {9798400712746},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3696410.3714747},
    booktitle = {Proceedings of the ACM on Web Conference 2025},
    pages = {35–46},
    numpages = {12}
    location = {Sydney, NSW, Australia},
    series = {WWW '25},
}
```
