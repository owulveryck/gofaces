package main

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

var (
	modelONNX     = "model.onnx"
	npyZeroResult = "testdata/zero_value_expected_output.npy"
)

// TestYOLO validate the exported model
func TestYOLO(t *testing.T) {
	b, err := ioutil.ReadFile(modelONNX)
	check(t, err)

	backend := gorgonnx.NewGraph()
	model := onnx.NewModel(backend)
	err = model.UnmarshalBinary(b)
	check(t, err)

	inputT := tensor.New(
		tensor.WithShape(1, 416, 416, 3),
		tensor.Of(tensor.Float32))

	err = model.SetInput(0, inputT)
	check(t, err)

	err = backend.Run()
	check(t, err)
	outputT, err := model.GetOutputTensors()
	check(t, err)

	file, err := os.Open(npyZeroResult)
	check(t, err)
	defer file.Close()
	expectedOutput := new(tensor.Dense)

	err = expectedOutput.ReadNpy(file)
	check(t, err)
	assert.Equal(t, expectedOutput.Shape(), outputT[0].Shape())

	outputValue := outputT[0].Data().([]float32)
	expectedValue := expectedOutput.Data().([]float32)

	assert.InDeltaSlice(t, expectedValue, outputValue, 5e-5)
}

func check(t *testing.T, err error) {
	if err != nil {
		t.Fatal(err)
	}
}
