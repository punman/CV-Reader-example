FROM python:3.8.7-slim

WORKDIR /

RUN apt update
RUN apt install python3-pip -y
RUN apt install git -y

RUN pip3 install six==1.12.0 -U
RUN pip3 install -U scikit-learn
RUN pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip3 install pyyaml==5.1 -U
RUN pip3 install Pillow
RUN pip3 install matplotlib
RUN pip3 install tensorflow
RUN pip3 install opencv-python
RUN pip3 install memory-profiler
RUN pip3 install python-Levenshtein
RUN apt install tesseract-ocr -y
RUN apt install libtesseract-dev -y
RUN pip3 install pytesseract
RUN pip3 install pandas
RUN pip3 install numpy==1.20.0 -U
RUN pip3 install streamlit
RUN pip3 install streamlit --upgrade
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install fastapi
RUN pip3 install uvicorn[standard]
RUN pip3 install jsonpickle
RUN apt-get install tesseract-ocr-tha -y

COPY . .
EXPOSE 8000
ENTRYPOINT ["uvicorn", "main:app" ,"--host", "0.0.0.0", "--port", "8000"]