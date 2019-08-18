package gofaces

// The WSize x HSize image is divided into a gridHeight x gridWidth grid.
// Each of these grid cells will predict boxesPerCell bounding boxes (see the Box structure)

const (
	// HSize is the height of the input picture
	HSize = 416
	// WSize is the width of the input picture
	WSize        = 416
	blockSize    = 32
	gridHeight   = 13
	gridWidth    = 13
	boxesPerCell = 5
	numClasses   = 1
)

var (
	// classes detected by the model; this package detects a single class
	classes = []string{"face"}
	anchors = []float64{0.738768, 0.874946, 2.42204, 2.65704, 4.30971, 7.04493, 10.246, 4.59428, 12.6868, 11.8741}
)
