package main

import (
	"flag"
	"fmt"
	"image/png"
	"io/ioutil"
	"log"
	"os"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/gofaces"
	"github.com/owulveryck/gofaces/draw"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

type configuration struct {
	ConfidenceThreshold float64 `envconfig:"confidence_threshold" default:"0.10" required:"true"`
	ClassProbaThreshold float64 `envconfig:"proba_threshold" default:"0.90" required:"true"`
}

var (
	model   = flag.String("model", "../model/model.onnx", "path to the model file")
	imgF    = flag.String("img", "../samples/olivier.wulveryck.jpg", "path of an input jpeg image")
	outputF = flag.String("output", "", "path of an output png file (empty means no file)")
	silent  = flag.Bool("s", false, "silent mode (useful if output is -)")
	config  configuration
)

func main() {
	err := envconfig.Process("yolo", &config)
	if err != nil {
		panic(err)
	}
	h := flag.Bool("h", false, "help")
	flag.Parse()
	if *h {
		err := envconfig.Usage("yolo", &config)
		if err != nil {
			panic(err)
		}
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

	boxes, err := gofaces.ProcessOutput(outputs[0].(*tensor.Dense))
	if err != nil {
		log.Fatal(err)
	}
	boxes = gofaces.Sanitize(boxes)
	if err != nil {
		log.Fatal(err)
	}

	for i := 1; i < len(boxes); i++ {
		if boxes[i].Confidence < config.ConfidenceThreshold {
			boxes = boxes[:i]
			//continue
		}
		/*
			if boxes[i].Classes[0].Prob < config.ClassProbaThreshold {
				boxes = boxes[:i]
				continue
			}
		*/
	}

	fmt.Println(boxes)

	if *outputF != "" {
		mask := draw.CreateMask(gofaces.WSize, gofaces.HSize, boxes)
		f, err := os.Create(*outputF)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		err = png.Encode(f, mask)
		if err != nil {
			log.Fatal(err)
		}
	}

}
