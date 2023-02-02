#!/bin/bash

# setup conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE"/etc/profile.d/conda.sh

# create conda env
read -rp "Enter environment name: " ENV_NAME
read -rp "Enter python version (default 3.10): " PYTHON_VERSION
if [ -z "$PYTHON_VERSION" ]; then
  PYTHON_VERSION="3.10"
fi
conda create -yn "$ENV_NAME" python="$PYTHON_VERSION"
conda activate "$ENV_NAME"

# replace placeholder env with $ENV_NAME in scripts/train.sh
NEW_CONDA_LINE="source \$CONDA_BASE/bin/activate $ENV_NAME"
sed -i '' "s,.*bin/activate.*,$NEW_CONDA_LINE,g" scripts/train.sh

# install torch
read -rp "Enter cuda version (e.g. '11.7', default no cuda support): " CUDA_VERSION
read -rp "Enter PyTorch version (e.g. '1.13.1', default latest): " PYTORCH_VERSION
if [ -n "$PYTORCH_VERSION" ]; then
  PYTORCH_VERSION="=$PYTORCH_VERSION"
fi
if [ -z "$CUDA_VERSION" ]; then
    echo conda install -y pytorch"$PYTORCH_VERSION" cpuonly -c pytorch
else
    echo conda install -y pytorch"$PYTORCH_VERSION" cudatoolkit="$CUDA_VERSION" -c pytorch
fi

# install python requirements
pip install -r requirements.txt

cd data/prepare_ontonotes
chmod 755 setup.sh
conda create -y --name py27 python=2.7

#!/bin/bash

eval "$(conda shell.bash hook)"

dlx() {
  wget $1/$2
  tar -xvzf $2
  rm $2
}

conll_url=http://conll.cemantix.org/2012/download
dlx $conll_url conll-2012-train.v4.tar.gz
dlx $conll_url conll-2012-development.v4.tar.gz
dlx $conll_url/test conll-2012-test-key.tar.gz
dlx $conll_url/test conll-2012-test-official.v9.tar.gz
dlx $conll_url conll-2012-scripts.v3.tar.gz

dlx http://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz
mv reference-coreference-scorers conll-2012/scorer

tar -xzvf ontonotes-release-5.0_LDC2013T19.tgz
ontonotes_path=ontonotes-release-5.0

conda activate py27
bash conll-2012/v3/scripts/skeleton2conll.sh -D $ontonotes_path/data/files/data conll-2012

function compile_partition() {
    rm -f $2.$5.$3$4
    cat conll-2012/$3/data/$1/data/$5/annotations/*/*/*/*.$3$4 >> $2.$5.$3$4
}

function compile_language() {
    compile_partition development dev v4 _gold_conll $1
    compile_partition train train v4 _gold_conll $1
    compile_partition test test v4 _gold_conll $1
}

compile_language english

conda activate coref
python minimize.py

