# codeaplaca-personified

Enhancing [Code Alpaca](https://github.com/sahil280114/codealpaca)-based instruction generation by applying synthetic programming personas from [argilla/FinePersonas-v0.1](https://huggingface.co/datasets/argilla/FinePersonas-v0.1).

## Installation

To install dependencies and the recommended LM serving framework (SGLang), run:

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Install SGLang from source for repro
git clone https://github.com/sgl-project/sglang.git
pushd sglang
git checkout 37c5899fc2100de1c9afd51a7b1977b2f8185a28
pip install -e "python[all]"
popd
# Install FlashInfer CUDA kernels
pip install flashinfer -U  -i https://flashinfer.ai/whl/cu121/torch2.4/
```