package gofaces

import "flag"

type configuration struct {
	ConfidenceThreshold float64 `envconfig:"confidence_threshold" default:"0.60" required:"true"`
	ClassProbaThreshold float64 `envconfig:"proba_threshold" default:"0.90" required:"true"`
}

// The 416x416 image is divided into a 13x13 grid. Each of these grid cells
// will predict 5 bounding boxes (boxesPerCell). A bounding box consists of
// five data items: x, y, width, height, and a confidence score. Each grid
// cell also predicts which class each bounding box belongs to.
//
const (
	hSize, wSize  = 416, 416
	blockSize     = 32
	gridHeight    = 13
	gridWidth     = 13
	boxesPerCell  = 5
	numClasses    = 1
	envConfPrefix = "yolo"
)

var (
	model       = flag.String("model", "model.onnx", "path to the model file")
	imgF        = flag.String("img", "", "path of an input jpeg image (use - for stdin)")
	outputF     = flag.String("output", "", "path of an output png file (use - for stdout)")
	silent      = flag.Bool("s", false, "silent mode (useful if output is -)")
	classes     = []string{"face"}
	anchors     = []float64{0.738768, 0.874946, 2.42204, 2.65704, 4.30971, 7.04493, 10.246, 4.59428, 12.6868, 11.8741}
	scaleFactor = float32(1) // The scale factor to resize the image to hSize*wSize
	config      configuration
)
