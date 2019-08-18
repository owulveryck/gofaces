package main

import (
	"flag"
	"io/ioutil"
	"log"
	"os"

	"github.com/owulveryck/gofaces"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
)

var (
	model   = flag.String("model", "../model/model.onnx", "path to the model file")
	imgF    = flag.String("img", "../samples/olivier.wulveryck.jpg", "path of an input jpeg image")
	outputF = flag.String("output", "", "path of an output png file (empty means no file)")
	silent  = flag.Bool("s", false, "silent mode (useful if output is -)")
)

func main() {
	h := flag.Bool("h", false, "help")
	flag.Parse()
	if *h {
		flag.Usage()
		os.Exit(0)
	}
	if _, err := os.Stat(*model); err != nil && os.IsNotExist(err) {
		log.Fatalf("%v does not exist", *model)
	}
	// Create a backend receiver
	backend := gorgonnx.NewGraph()
	// Create a model and set the execution backend
	m := onnx.NewModel(backend)

	// read the onnx model
	b, err := ioutil.ReadFile(*model)
	if err != nil {
		log.Fatal(err)
	}
	// Decode it into the model
	err = m.UnmarshalBinary(b)
	if err != nil {
		log.Fatal(err)
	}

	img, err := os.Open(*imgF)
	if err != nil {
		log.Fatal(err)
	}
	inputT, err := gofaces.GetTensorFromImage(img)
	if err != nil {
		log.Fatal(err)
	}
	m.SetInput(0, inputT)
	err = backend.Run()
	if err != nil {
		log.Fatal(err)
	}
	outputs, err := m.GetOutputTensors()

	log.Println(outputs[0])

}
