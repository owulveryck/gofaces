package gofaces

import (
	"image"
	"testing"

	"gorgonia.org/tensor"
)

func TestImageToNormalizeBWHC(t *testing.T) {
	// Create a blank image 10 pixels wide by 4 pixels tall
	myImage := image.NewRGBA(image.Rect(0, 0, 10, 10))

	for x := 0; x < 10; x++ {
		for y := 0; y < 10; y++ {
			offset := myImage.PixOffset(x, y)
			myImage.Pix[offset+3] = 255 // 1st pixel alpha
		}
	}

	// draw 3 rectangles 5x5; one red, one green, one blue
	for x := 0; x < 5; x++ {
		for y := 0; y < 5; y++ {
			offset := myImage.PixOffset(x, y)
			myImage.Pix[offset+0] = 255 // 1st pixel red
			myImage.Pix[offset+1] = 0   // 1st pixel green
			myImage.Pix[offset+2] = 0   // 1st pixel blue
			myImage.Pix[offset+3] = 255 // 1st pixel alpha
		}
	}
	for x := 0; x < 5; x++ {
		for y := 5; y < 10; y++ {
			offset := myImage.PixOffset(x, y)
			myImage.Pix[offset+0] = 0   // 1st pixel red
			myImage.Pix[offset+1] = 255 // 1st pixel green
			myImage.Pix[offset+2] = 0   // 1st pixel blue
			myImage.Pix[offset+3] = 255 // 1st pixel alpha
		}
	}
	for x := 5; x < 10; x++ {
		for y := 5; y < 10; y++ {
			offset := myImage.PixOffset(x, y)
			myImage.Pix[offset+0] = 0   // 1st pixel red
			myImage.Pix[offset+1] = 0   // 1st pixel green
			myImage.Pix[offset+2] = 255 // 1st pixel blue
			myImage.Pix[offset+3] = 255 // 1st pixel alpha
		}
	}
	dense := tensor.NewDense(
		tensor.Float32,
		[]int{1, 10, 10, 3})
	err := imageToNormalizedBWHC(myImage, dense)
	if err != nil {
		t.Fatal(err)
	}
	ret, err := dense.Sum(0, 1, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if ret.Data().(float32) != 75 {
		t.Fail()
	}
}
