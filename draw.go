package gofaces

import (
	"fmt"
	"image"
	"image/color"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

var col = color.RGBA{255, 0, 0, 255} // Red
// drawRectangle r onto img and add a label
func drawRectangle(img *image.NRGBA, r image.Rectangle, label string) {

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

	// Rect draws a rectangle utilizing HLine() and VLine()
	rect := func(r image.Rectangle) {
		hLine(r.Min.X, r.Max.Y, r.Max.X)
		hLine(r.Min.X, r.Max.Y, r.Max.X)
		hLine(r.Min.X, r.Min.Y, r.Max.X)
		vLine(r.Max.X, r.Min.Y, r.Max.Y)
		vLine(r.Min.X, r.Min.Y, r.Max.Y)
	}
	drawLabel(img, r.Min.X+5, r.Min.Y+15, label)
	rect(r)
}

// drawLabel on img at pos x, y
func drawLabel(img *image.NRGBA, x, y int, label string) {
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

// createMask returns an image with the boxes and labels
func createMask(width, height int, boxes []box) image.Image {
	m := image.NewNRGBA(image.Rect(0, 0, width, height))

	for _, b := range boxes {
		drawRectangle(m, b.r, fmt.Sprintf("%v %2.2f%%", b.classes[0].class, b.classes[0].prob*100))
	}
	return m
}
