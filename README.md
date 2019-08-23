# gofaces
This repository is an implementation of this [article](http://blog.owulveryck.info/2019/08/16/a-simple-face-detection-utility-from-python-to-go.html).

It is a toy to run face detection on a picture with a neural network.

The neural network is based on Tiny YOLO v2 and is encoded in the onnx format.
It decocded with [onnx-go](https://github.com/owulveryck/onnx-go) and executed by [Gorgonia](https://github.com/gorgonia/gorgonia).

# Installation an usage
## Requirements
* Go with modules support, `1.12+` recommended.
* [Git LFS](https://git-lfs.github.com). It's needed in order to get the file model.onnx when cloning the repo. If you don’t have this, running will fail with an error: proto: can’t skip unknown wire type 6

## Usage
```
git clone https://github.com/owulveryck/gofaces
cd gofaces
git lfs install
git lfs fetch
git lfs checkout model/model.onnx
cd cmd
go run main.go -h
```
