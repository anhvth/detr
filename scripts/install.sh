conda install -c pytorch pytorch torchvision
conda install cython scipy -y
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install git+https://github.com/cocodataset/panopticapi.git

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

pip install -r requirements/build.txt
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install -v -e .  # or "python setup.py develop"