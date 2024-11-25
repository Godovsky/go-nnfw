package main

import (
	"fmt"
	"nn/nn"
)

func main() {
	var nn nn.NeuralNetwork
	data_path := "data.csv"
	err := nn.Constructor(data_path, 2, 6, 6)
	if err != nil {
		fmt.Println(err)
		return
	}

	nn.Offset = 0.1
	nn.Step = 0.001

	for i := 0; i < 200000; i++ {
		nn.Train()
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			nn.Inputs[0] = float32(i)
			nn.Inputs[1] = float32(j)
			nn.Calculate()
			nn.Print()
		}
	}
}
