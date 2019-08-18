package gofaces

import (
	"math"
)

func sigmoid(sum float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-sum))))
}
func sigmoid64(sum float32) float64 {
	return 1.0 / (1.0 + math.Exp(float64(-sum)))
}
func exp(val float32) float64 {
	return math.Exp(float64(val))
}

func softmax(a []float32) []float64 {
	var sum float64
	output := make([]float64, len(a))

	for i := 0; i < len(a); i++ {
		output[i] = math.Exp(float64(a[i]))
		sum += output[i]
	}
	for i := 0; i < len(output); i++ {
		output[i] = output[i] / sum
	}
	return output
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
