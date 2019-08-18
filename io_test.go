package gofaces

import (
	"bytes"
	"image"
	"image/jpeg"
	"testing"
)

func TestReadIMG(t *testing.T) {
	myImage := image.NewRGBA(image.Rect(0, 0, 500, 600))
	io := new(bytes.Buffer)
	err := jpeg.Encode(io, myImage, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = readIMG(io)
	if err != nil {
		t.Fatal(err)
	}

}
