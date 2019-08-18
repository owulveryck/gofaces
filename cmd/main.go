package main

import (
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"io"
	"io/ioutil"
	"log"
	"math"
	"os"
	"reflect"
	"sort"

	"github.com/disintegration/gift"
	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

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

type configuration struct {
	ConfidenceThreshold float64 `envconfig:"confidence_threshold" default:"0.60" required:"true"`
	ClassProbaThreshold float64 `envconfig:"proba_threshold" default:"0.90" required:"true"`
}

func init() {
	err := envconfig.Process(envConfPrefix, &config)
	if err != nil {
		panic(err)
	}
}

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

func main() {
	h := flag.Bool("h", false, "help")
	flag.Parse()
	if *h {
		flag.Usage()
		envconfig.Usage(envConfPrefix, &config)
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
	must(m.UnmarshalBinary(b))

	input := getInput()
	m.SetInput(0, input)
	must(backend.Run())
	outputs, err := m.GetOutputTensors()

	processOutput(outputs, err)

}

func readIMG() (image.Image, error) {
	if *imgF == "" {
		flag.Usage()
		os.Exit(1)
	}
	var f io.Reader
	var err error
	if *imgF == "-" {
		f = os.Stdin
	} else {
		f, err = os.Open(*imgF)
		if err != nil {
			log.Fatal(err)
		}
		defer f.(*os.File).Close()
	}
	return jpeg.Decode(f)
}

func getInput() tensor.Tensor {
	img, err := readIMG()
	var resizeFilter gift.Filter
	if (img.Bounds().Max.X - img.Bounds().Min.X) > (img.Bounds().Max.Y - img.Bounds().Min.Y) {
		scaleFactor = float32(img.Bounds().Max.Y-img.Bounds().Min.Y) / float32(hSize)
		resizeFilter = gift.Resize(0, hSize, gift.LanczosResampling)
	} else {
		scaleFactor = float32(img.Bounds().Max.X-img.Bounds().Min.X) / float32(wSize)
		resizeFilter = gift.Resize(wSize, 0, gift.LanczosResampling)
	}

	inputT := tensor.New(tensor.WithShape(1, wSize, hSize, 3), tensor.Of(tensor.Float32))

	filters := []gift.Filter{
		resizeFilter,
		gift.CropToSize(wSize, hSize, gift.LeftAnchor),
		gift.Colorize(1.5, 0.1, 100),
	}
	for _, filter := range filters {
		g := gift.New(filter)
		dst := image.NewNRGBA(image.Rect(0, 0, wSize, hSize))
		g.Draw(dst, img)
		img = dst
	}
	err = imageToBWHC(img, inputT)
	if err != nil {
		log.Fatal(err)
	}
	return inputT
}

func processOutput(t []tensor.Tensor, err error) {
	if err != nil {
		log.Fatal(err)
	}
	dense := t[0].(*tensor.Dense)
	must(dense.Reshape(gridHeight, gridWidth, (5+numClasses)*boxesPerCell))
	data, err := native.Tensor3F32(dense)
	if err != nil {
		log.Fatal(err)
	}

	var boxes = make([]box, gridHeight*gridWidth*boxesPerCell)
	var counter int
	// https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/yolo_layer.c#L159
	for cx := 0; cx < gridWidth; cx++ {
		for cy := 0; cy < gridHeight; cy++ {
			for b := 0; b < boxesPerCell; b++ {
				channel := b * (numClasses + 5)
				tx := data[cy][cx][channel]
				ty := data[cy][cx][channel+1]
				tw := data[cy][cx][channel+2]
				th := data[cy][cx][channel+3]
				tc := data[cy][cx][channel+4]
				tclasses := make([]float32, numClasses)
				for i := 0; i < numClasses; i++ {
					tclasses[i] = data[cy][cx][channel+5+i]
				}

				// The predicted tx and ty coordinates are relative to the location
				// of the grid cell; we use the logistic sigmoid to constrain these
				// coordinates to the range 0 - 1. Then we add the cell coordinates
				// (0-12) and multiply by the number of pixels per grid cell (32).
				// Now x and y represent center of the bounding box in the original
				// 416x416 image space.
				// https://github.com/hollance/Forge/blob/04109c856237faec87deecb55126d4a20fa4f59b/Examples/YOLO/YOLO/YOLO.swift#L154
				x := int((float32(cx) + sigmoid(tx)) * blockSize)
				y := int((float32(cy) + sigmoid(ty)) * blockSize)
				// The size of the bounding box, tw and th, is predicted relative to
				// the size of an "anchor" box. Here we also transform the width and
				// height into the original 416x416 image space.
				w := int(exp(tw) * anchors[2*b] * blockSize)
				h := int(exp(th) * anchors[2*b+1] * blockSize)

				boxes[counter] = box{
					gridcell:   []int{cx, cy},
					r:          image.Rect(max(y-w/2, 0), max(x-h/2, 0), min(y+w/2, wSize), min(x+h/2, hSize)),
					confidence: sigmoid64(tc),
					classes:    getOrderedElements(softmax(tclasses)),
				}
				counter++
			}
		}
	}
	boxes = sanitize(boxes)
	if !*silent {
		printClassification(boxes)
	}
	if *outputF != "" {
		drawClassification(boxes)
	}
}

func printClassification(boxes []box) {
	var elements []element
	for _, box := range boxes {
		if box.classes[0].prob > config.ConfidenceThreshold {
			elements = append(elements, box.classes...)
			fmt.Printf("at (%v) with confidence %2.2f%%: %v\n", box.r, box.confidence, box.classes)
		}
	}
	sort.Sort(sort.Reverse(byProba(elements)))
	for _, c := range elements {
		if c.prob > 0.4 {
			fmt.Println(c)
		}
	}

}
func drawClassification(boxes []box) {
	if *outputF == "" {
		return
	}
	var f io.Writer
	var err error
	if *outputF == "-" {
		f = os.Stdout
	} else {
		f, err = os.Create(*outputF)
		if err != nil {
			log.Fatal(err)
		}
		defer f.(*os.File).Close()
	}
	img, err := readIMG()
	if err != nil {
		log.Fatal(err)
	}
	m := image.NewNRGBA(img.Bounds())

	draw.Draw(m, m.Bounds(), img, image.ZP, draw.Src)
	for _, b := range boxes {
		drawRectangle(m, b.r, fmt.Sprintf("%v %2.2f%%", b.classes[0].class, b.classes[0].prob*100))
	}

	if err := png.Encode(f, m); err != nil {
		log.Fatal(err)
	}

}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

type element struct {
	prob  float64
	class string
}

type byProba []element

func (b byProba) Len() int           { return len(b) }
func (b byProba) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byProba) Less(i, j int) bool { return b[i].prob < b[j].prob }

type box struct {
	r          image.Rectangle
	gridcell   []int
	confidence float64
	classes    []element
}

type byConfidence []box

func (b byConfidence) Len() int           { return len(b) }
func (b byConfidence) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byConfidence) Less(i, j int) bool { return b[i].confidence < b[j].confidence }

func sigmoid(sum float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-sum))))
}
func sigmoid64(sum float32) float64 {
	return 1.0 / (1.0 + math.Exp(float64(-sum)))
}
func exp(val float32) float64 {
	return math.Exp(float64(val))
}

func softmax(a []float32) []float64 {
	var sum float64
	output := make([]float64, len(a))

	for i := 0; i < len(a); i++ {
		output[i] = math.Exp(float64(a[i]))
		sum += output[i]
	}
	for i := 0; i < len(output); i++ {
		output[i] = output[i] / sum
	}
	return output
}

func getOrderedElements(input []float64) []element {
	elems := make([]element, len(input))
	for i := 0; i < len(elems); i++ {
		elems[i] = element{
			prob:  input[i],
			class: classes[i],
		}
	}
	sort.Sort(sort.Reverse(byProba(elems)))
	return elems
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func drawRectangle(img *image.NRGBA, r image.Rectangle, label string) {
	col := color.RGBA{255, 0, 0, 255} // Red

	// HLine draws a horizontal line
	hLine := func(x1, y, x2 int) {
		for ; x1 <= x2; x1++ {
			img.Set(x1, y, col)
		}
	}

	// VLine draws a veritcal line
	vLine := func(x, y1, y2 int) {
		for ; y1 <= y2; y1++ {
			img.Set(x, y1, col)
		}
	}

	minX := int(float32(r.Min.X) * scaleFactor)
	maxX := int(float32(r.Max.X) * scaleFactor)
	minY := int(float32(r.Min.Y) * scaleFactor)
	maxY := int(float32(r.Max.Y) * scaleFactor)
	// Rect draws a rectangle utilizing HLine() and VLine()
	rect := func(r image.Rectangle) {
		hLine(minX, maxY, maxX)
		hLine(minX, maxY, maxX)
		hLine(minX, minY, maxX)
		vLine(maxX, minY, maxY)
		vLine(minX, minY, maxY)
	}
	addLabel(img, minX+5, minY+15, label)
	rect(r)
}

func addLabel(img *image.NRGBA, x, y int, label string) {
	col := color.NRGBA{0, 255, 0, 255}
	point := fixed.Point26_6{
		X: fixed.Int26_6(x * 64),
		Y: fixed.Int26_6(y * 64),
	}

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(col),
		Face: basicfont.Face7x13,
		Dot:  point,
	}
	d.DrawString(label)
}

// from https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
// 1- Sort the predictions by the confidence scores.
// 2- Start from the top scores, ignore any current prediction if we find any previous predictions that have the same class and IoU > 0.5 with the current prediction.
// 3- Repeat step 2 until all predictions are checked.
func sanitize(boxes []box) []box {
	sort.Sort(sort.Reverse(byConfidence(boxes)))

	for i := 1; i < len(boxes); i++ {
		if boxes[i].confidence < config.ConfidenceThreshold {
			boxes = boxes[:i]
			break
		}
		if boxes[i].classes[0].prob < config.ClassProbaThreshold {
			boxes = boxes[:i]
			break
		}
		for j := i + 1; j < len(boxes); {
			iou := iou(boxes[i].r, boxes[j].r)
			if iou > 0.5 && boxes[i].classes[0].class == boxes[j].classes[0].class {
				boxes = append(boxes[:j], boxes[j+1:]...)
				continue
			}
			j++
		}
	}
	return boxes
}

// evaluate the intersection over union of two rectangles
func iou(r1, r2 image.Rectangle) float64 {
	// get the intesection rectangle
	intersection := image.Rect(
		max(r1.Min.X, r2.Min.X),
		max(r1.Min.Y, r2.Min.Y),
		min(r1.Max.X, r2.Max.X),
		min(r1.Max.Y, r2.Max.Y),
	)
	// compute the area of intersection rectangle
	interArea := area(intersection)
	r1Area := area(r1)
	r2Area := area(r2)
	// compute the intersection over union by taking the intersection
	// area and dividing it by the sum of prediction + ground-truth
	// areas - the interesection area
	return float64(interArea) / float64(r1Area+r2Area-interArea)
}

func area(r image.Rectangle) int {
	return max(0, r.Max.X-r.Min.X-1) * max(0, r.Max.Y-r.Min.Y-1)
}

// ImageToBWHC convert an image to a BWHC tensor
// this function returns an error if:
//
//   - dst is not a pointer
//   - dst's shape is not 4
//   - dst' second dimension is not 1
//   - dst's third dimension != i.Bounds().Dy()
//   - dst's fourth dimension != i.Bounds().Dx()
//   - dst's type is not float32 or float64 (temporarly)
func imageToBWHC(img image.Image, dst tensor.Tensor) error {
	// check if tensor is a pointer
	rv := reflect.ValueOf(dst)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return errors.New("cannot decode image into a non pointer or a nil receiver")
	}
	// check if tensor is compatible with BWHC (4 dimensions)
	if len(dst.Shape()) != 4 {
		return fmt.Errorf("Expected a 4 dimension tensor, but receiver has only %v", len(dst.Shape()))
	}
	// Check the batch size
	if dst.Shape()[0] != 1 {
		return errors.New("only batch size of one is supported")
	}
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	if dst.Shape()[1] != h || dst.Shape()[2] != w {
		return fmt.Errorf("cannot fit image into tensor; image is %v*%v but tensor is %v*%v", h, w, dst.Shape()[2], dst.Shape()[3])
	}
	switch dst.Dtype() {
	case tensor.Float32:
		for x := 0; x < w; x++ {
			for y := 0; y < h; y++ {
				r, g, b, a := img.At(x, y).RGBA()
				if a != 65535 {
					return errors.New("transparency not handled")
				}
				err := dst.SetAt(float32(r)/65535, 0, x, y, 1)
				if err != nil {
					return err
				}
				err = dst.SetAt(float32(g)/65535, 0, x, y, 1)
				if err != nil {
					return err
				}
				err = dst.SetAt(float32(b)/65535, 0, x, y, 2)
				if err != nil {
					return err
				}
			}
		}
	default:
		return fmt.Errorf("%v not handled yet", dst.Dtype())
	}
	return nil

}
