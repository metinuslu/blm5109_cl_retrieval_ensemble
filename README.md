# Usage of Some Embedding Algorithms (Text to Embedding) and Retrieval Ensemble for BLM5109 Collective Learning Course at YTU

## Project Details
> **Description:** Usage of Some Embedding Algorithms (Text to Embedding) and Retrieval Ensemble on Text Dataset & More Details: KolektifOgrenme_HW3_2024Guz.pdf  
> **Course Name:** BLM5109 - Collective Learning  
> **Course Url:** http://bologna.yildiz.edu.tr/index.php?r=course/view&id=6047&aid=3  
> **Course Page:** https://sites.google.com/view/mfatihamasyali/kolektif-%C3%B6%C4%9Frenme  

## Installation
**Step-1:** Create Env. 
```
conda create --name "py_env_312" python=3.12  
conda activate py_env_312  
python -V
```

**Step-2:** Install Libraries & Frameworks
```
# https://jupyter.org/install
pip install jupyterlab

# https://www.tensorflow.org/install
pip install tensorflow

# https://pytorch.org/
pip install torch torchvision torchaudio

# Install Requirements File
pip install -r requirements.txt
```

**Step-3:** Start Jupyter IDE
```
# Start Jupyter Lab / Notebook
jupyter lab
jupyter notebook
```

## Datasets
    - Turkish Instructions: https://huggingface.co/datasets/merve/turkish_instructions  

## Preprocess & Modelling
  - Dataset MetaData
    - Columns: Soru[Talimat, Giris], Cevap
    - Shape: 2 x 2000

  - Embedding Models
    - [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
    - [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)
    - [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
    - [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
    - [ytu-ce-cosmos/turkish-colbert](https://huggingface.co/ytu-ce-cosmos/turkish-colbert)
    - [thenlper/gte-large](https://huggingface.co/thenlper/gte-large)

  - Vector Distance
    - Cosine Similarity
    - ![Cosine Similarity](https://storage.googleapis.com/lds-media/images/cosine-similarity-vectors.original.jpg "Cosine Similarity")
  - Embedding Accuracy
    - Top-1
    - Top-5

  - Retrieval Ensemble
    - Ret. Ens. - 1: Majority Voting
    - Ret. Ens. - 2: Average Cosine 
    - Ret. Ens. - 3: Weighted Average Cosine

  - Turkish Instructions
    - Embedding Code: Retrieval_Embedding-v{1..2}.ipynb
    - Ensemble Code: Retrieval_Ensemble-v3{1..2}.ipynb

## Contact
    - Ahmed Ugur - 23501027  
    - Metin Uslu - 235B7014