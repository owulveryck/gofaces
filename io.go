package gofaces

import (
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"sort"
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
