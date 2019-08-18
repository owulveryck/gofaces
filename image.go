package gofaces

import (
	"errors"
	"fmt"
	"image"
	"reflect"

	"github.com/disintegration/gift"
	"gorgonia.org/tensor"
)

// getSuitableImage from the io.Reader reads the image and do the pre-processing such
// as resize, crop or saturation
func resizeImage(img image.Image) (*image.RGBA, error) {
	width := img.Bounds().Max.X - img.Bounds().Min.X
	height := img.Bounds().Max.Y - img.Bounds().Min.Y
	resizeFilter := gift.Resize(WSize, 0, gift.LanczosResampling)
	if height > width {
		resizeFilter = gift.Resize(0, HSize, gift.LanczosResampling)
	}
	dst := image.NewRGBA(image.Rect(0, 0, WSize, HSize))
	gift.New(resizeFilter).Draw(dst, img)
	return dst, nil
}

// Create a tensor BHWC from the image; the values are normalized between 0 and 1
func imageToNormalizedBWHC(img *image.RGBA, dst tensor.Tensor) error {
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
				err := dst.SetAt(float32(r)/65535, 0, x, y, 0)
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
