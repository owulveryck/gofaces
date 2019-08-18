package gofaces

import (
	"testing"
)

func TestMin(t *testing.T) {
	if min(1, 0) != 0 {
		t.Fail()
	}
	if min(0, 1) != 0 {
		t.Fail()
	}
}
func TestMax(t *testing.T) {
	if max(1, 0) != 1 {
		t.Fail()
	}
	if max(0, 1) != 1 {
		t.Fail()
	}
}

func TestSoftmax(t *testing.T) {
	input := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	expected := []float64{0.0005766127696870058,
		0.0015673960138976283,
		0.004260624102577064,
		0.01158157707592986,
		0.03148199051039798,
		0.08557692272813494,
		0.23262219398733308,
		0.6323326828120425}
	output := softmax(input)
	for i := 0; i < len(output); i++ {
		if expected[i] != output[i] {
			t.Fail()
		}
	}
}
