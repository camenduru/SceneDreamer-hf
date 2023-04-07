export CUDA_VERSION=$(nvcc --version| grep -Po "(\d+\.)+\d+" | head -1)
CURRENT=$(pwd)
for p in correlation channelnorm resample2d bias_act upfirdn2d; do
    cd imaginaire/third_party/${p};
    rm -rf build dist *info;
    python setup.py install;
    cd ${CURRENT};
done

for p in gancraft/voxlib; do
  cd imaginaire/model_utils/${p};
  make all
  python -m pip install .
  cd ${CURRENT};
done

cd gridencoder
python setup.py build_ext --inplace
python -m pip install .
cd ${CURRENT}