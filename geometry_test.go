package gofaces

import (
	"image"
	"testing"
)

func TestArea(t *testing.T) {
	res := area(image.Rect(0, 0, 5, 5))
	if res != 25 {
		t.Fatal(res)
	}
	res = area(image.Rect(1, 2, 6, 7))
	if res != 25 {
		t.Fatal(res)
	}

}

func TestIOU(t *testing.T) {
	rect1 := image.Rect(0, 0, 5, 5)
	rect2 := image.Rect(1, 1, 6, 6)
	res := iou(rect1, rect2)
	if res != float64(16.0/(50.0-16.0)) {
		t.Fatal(res)
	}
	res = iou(rect2, rect1)
	if res != float64(16.0/(50.0-16.0)) {
		t.Fatal(res)
	}
}
