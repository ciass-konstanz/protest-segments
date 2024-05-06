#!/usr/bin/env bash

echo "Installing dependencies"

# install requirements
pip install --no-cache-dir --upgrade -r ./requirements.txt

# install detectron2
pip install -e ./modules/detectron2

# check if cuda is available
cuda=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$cuda" == "True" ]; then
  # install MaskDINO
  cd ./modules/MaskDINO
  pip install -r requirements.txt
  cd maskdino/modeling/pixel_decoder/ops
  python setup.py build install --install-lib $HOME/.local/lib/python3.8/site-packages/
  cd ../../../../../../

  # install Detic
  cd ./modules/Detic
  pip install -r requirements.txt
  cd third_party/Deformable-DETR/models/ops
  python setup.py build install --install-lib $HOME/.local/lib/python3.8/site-packages/
  cd ../../../../../../

else
  echo "Could not find graphics card... skipping dependencies..."
fi
