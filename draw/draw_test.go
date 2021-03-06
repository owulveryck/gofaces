package draw

import (
	"image"
	"testing"

	"github.com/owulveryck/gofaces"
)

func TestCreateMask(t *testing.T) {
	boxes := []gofaces.Box{
		gofaces.Box{
			R: image.Rect(1, 1, 6, 6),
			Elements: []gofaces.Element{
				gofaces.Element{
					Prob:  0.6,
					Class: "test",
				},
			},
		},
	}
	mask := CreateMask(416, 416, boxes)
	cR, cG, cB, cA := col.RGBA()
	for x := 1; x < 7; x++ {
		r, g, b, a := mask.At(x, 1).RGBA()
		if r != cR || g != cG || b != cB || a != cA {
			t.Fail()
		}
		r, g, b, a = mask.At(x, 6).RGBA()
		if r != cR || g != cG || b != cB || a != cA {
			t.Fail()
		}
	}
	for y := 1; y < 7; y++ {
		r, g, b, a := mask.At(1, y).RGBA()
		if r != cR || g != cG || b != cB || a != cA {
			t.Fail()
		}
		r, g, b, a = mask.At(6, y).RGBA()
		if r != cR || g != cG || b != cB || a != cA {
			t.Fail()
		}
	}

}
