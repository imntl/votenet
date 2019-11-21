FROM pytorch/pytorch:latest

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip install mlflow
RUN pip install --verbose --no-cache-dir torch-scatter
RUN pip install --verbose --no-cache-dir torch-sparse
RUN pip install --verbose --no-cache-dir torch-cluster
RUN pip install --verbose --no-cache-dir torch-spline-conv
RUN pip install torch-geometric
RUN pip install tensorflow
RUN pip install future
RUN pip install matplotlib
RUN pip install opencv-python
RUN pip install open3d-python
RUN pip install scikit-image
RUN pip install matplotlib
RUN pip install plyfile
RUN pip install 'trimesh>=2.35.39,<2.35.40'
RUN pip install git+git://github.com/erikwijmans/etw_pytorch_utils.git@v1.1.1#egg=etw_pytorch_utils \
			h5py \
			numpy \
			pprint \
			enum34 \
			future 
RUN apt update && apt install -y libglib2.0-dev libsm6 libxrender-dev libxext-dev locales

EXPOSE 5000

CMD ["bash"]
