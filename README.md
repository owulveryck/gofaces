# gofaces

This repository is an implementation of this [article](http://blog.owulveryck.info/2019/08/16/a-simple-face-detection-utility-from-python-to-go.html).

It is a toy to run face detection on a picture with a neural network.

The neural network is based on Tiny YOLO v2 and is encoded in the onnx format. 
It decocded with [onnx-go](https://github.com/owulveryck/onnx-go) and executed by [Gorgonia](https://github.com/gorgonia/gorgonia).

# Installation an usage

```
go get github.com/owulveryck/gofaces
cd $GOPATH/src/github.com/owulveryck/gofaces/cmd
go run main.go -h
```
