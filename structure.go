package gofaces

import "image"

type element struct {
	prob  float64
	class string
}

type box struct {
	r          image.Rectangle
	gridcell   []int
	confidence float64
	classes    []element
}
