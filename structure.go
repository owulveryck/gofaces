package gofaces

import (
	"bytes"
	"fmt"
	"image"
)

// Element in a box
type Element struct {
	Prob  float64
	Class string
}

// Box is holding a bounding box
// A bunding box is a rectangle R containing an object with Confidence.
// The object is one of the Elements (most likely the one with the highest probability)
type Box struct {
	R          image.Rectangle
	gridcell   []int
	Confidence float64 // The confidence the model has that there is at least one element in this box
	Elements   []Element
}

func (b Box) String() string {
	buf := new(bytes.Buffer)
	fmt.Fprintf(buf, "At %v (confidence %2.2f):\n", b.R, b.Confidence)
	for _, elem := range b.Elements {
		fmt.Fprintf(buf, "\t- %v - %v\n", elem.Class, elem.Prob)
	}
	return buf.String()
}
