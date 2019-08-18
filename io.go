package gofaces

import (
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"sort"

	"gorgonia.org/tensor"
)

func readIMG(r io.Reader) (image.Image, error) {
	return jpeg.Decode(r)
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

// GetTensorFromImage reads an image from r and returns a tensor suitable to run in tiny yolo.
// The tensor is BWHC and is normalized;
// its shape is (1,wSize,hSize,3)
func GetTensorFromImage(r io.Reader) (tensor.Tensor, error) {
	img, err := readIMG(r)
	if err != nil {
		return nil, err
	}
	resized, err := resizeImage(img)
	if err != nil {
		return nil, err
	}
	t := tensor.NewDense(tensor.Float32, []int{1, wSize, hSize, 3})
	err = imageToNormalizedBWHC(resized, t)
	return t, err
}
