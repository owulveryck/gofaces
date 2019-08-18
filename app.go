package gofaces

import (
	"image"
	"log"
	"sort"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

// ProcessOutput analyze the tensor t and output the prediction boxes
func ProcessOutput(dense *tensor.Dense) ([]Box, error) {

	err := dense.Reshape(gridHeight, gridWidth, (5+numClasses)*boxesPerCell)
	if err != nil {
		return nil, err
	}
	data, err := native.Tensor3F32(dense)
	if err != nil {
		log.Fatal(err)
	}

	var boxes = make([]Box, gridHeight*gridWidth*boxesPerCell)
	var counter int
	// https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/yolo_layer.c#L159
	for cx := 0; cx < gridWidth; cx++ {
		for cy := 0; cy < gridHeight; cy++ {
			for b := 0; b < boxesPerCell; b++ {
				channel := b * (numClasses + 5)
				tx := data[cy][cx][channel]
				ty := data[cy][cx][channel+1]
				tw := data[cy][cx][channel+2]
				th := data[cy][cx][channel+3]
				tc := data[cy][cx][channel+4]
				tclasses := make([]float32, numClasses)
				for i := 0; i < numClasses; i++ {
					tclasses[i] = data[cy][cx][channel+5+i]
				}

				// The predicted tx and ty coordinates are relative to the location
				// of the grid cell; we use the logistic sigmoid to constrain these
				// coordinates to the range 0 - 1. Then we add the cell coordinates
				// (0-12) and multiply by the number of pixels per grid cell (32).
				// Now x and y represent center of the bounding box in the original
				// 416x416 image space.
				// https://github.com/hollance/Forge/blob/04109c856237faec87deecb55126d4a20fa4f59b/Examples/YOLO/YOLO/YOLO.swift#L154
				x := int((float32(cx) + sigmoid(tx)) * blockSize)
				y := int((float32(cy) + sigmoid(ty)) * blockSize)
				// The size of the bounding box, tw and th, is predicted relative to
				// the size of an "anchor" box. Here we also transform the width and
				// height into the original 416x416 image space.
				w := int(exp(tw) * anchors[2*b] * blockSize)
				h := int(exp(th) * anchors[2*b+1] * blockSize)

				boxes[counter] = Box{
					gridcell:   []int{cx, cy},
					R:          image.Rect(max(y-w/2, 0), max(x-h/2, 0), min(y+w/2, wSize), min(x+h/2, hSize)),
					Confidence: sigmoid64(tc),
					Classes:    getOrderedElements(softmax(tclasses)),
				}
				counter++
			}
		}
	}
	return boxes, nil
}

// Sanitize the output
// from https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
//
// - Sort the predictions by the confidence scores.
//
// - Start from the top scores, ignore any current prediction if we find any previous predictions that have the same class and IoU > 0.5 with the current prediction.
//
// - Repeat step 2 until all predictions are checked.
func Sanitize(boxes []Box) []Box {
	sort.Sort(sort.Reverse(byConfidence(boxes)))

	for i := 1; i < len(boxes); i++ {
		for j := i + 1; j < len(boxes); {
			iou := iou(boxes[i].R, boxes[j].R)
			if iou > 0.5 && boxes[i].Classes[0].Class == boxes[j].Classes[0].Class {
				boxes = append(boxes[:j], boxes[j+1:]...)
				continue
			}
			j++
		}
	}
	return boxes
}
