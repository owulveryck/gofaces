package gofaces

import (
	"image"
	"image/jpeg"
	"io"

	"gorgonia.org/tensor"
)

func readIMG(r io.Reader) (image.Image, error) {
	return jpeg.Decode(r)
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
	t := tensor.NewDense(tensor.Float32, []int{1, WSize, HSize, 3})
	err = imageToNormalizedBWHC(resized, t)
	return t, err
}
