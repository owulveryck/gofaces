package gofaces

import "image"

// evaluate the intersection over union of two rectangles
func iou(r1, r2 image.Rectangle) float64 {
	// get the intesection rectangle
	intersection := image.Rect(
		max(r1.Min.X, r2.Min.X),
		max(r1.Min.Y, r2.Min.Y),
		min(r1.Max.X, r2.Max.X),
		min(r1.Max.Y, r2.Max.Y),
	)
	// compute the area of intersection rectangle
	interArea := area(intersection)
	r1Area := area(r1)
	r2Area := area(r2)
	// compute the intersection over union by taking the intersection
	// area and dividing it by the sum of prediction + ground-truth
	// areas - the interesection area
	return float64(interArea) / float64(r1Area+r2Area-interArea)
}
func area(r image.Rectangle) int {
	return max(0, r.Max.X-r.Min.X) * max(0, r.Max.Y-r.Min.Y)
}
