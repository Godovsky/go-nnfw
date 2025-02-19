package main

import (
	"fmt"
	"nn/nn"
)

func main() {
	var n nn.NeuralNetwork

	err := n.LoadFromJson("parameters.json")
	if err != nil {
		fmt.Println(err)

		err = n.Constructor(2, 4, 6)
		if err != nil {
			fmt.Println(err)
			return
		}

		n.Offset = 0.01
		n.Step = 0.01
	}

	err = n.GetDataFromFile("data.csv")
	if err != nil {
		fmt.Println(err)
		return
	}

	// You can add the data line
	// err = n.AddDataLine(1, 1, 1, 1, 0, 0, 0, 1)
	// if err != nil {
	// 	fmt.Println(err)
	// 	return
	// }
	// fmt.Println(n.TrainData)

	for i := 0; i < 100000; i++ {
		n.Train()
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			n.Inputs[0] = float32(i)
			n.Inputs[1] = float32(j)
			n.Calculate()
			n.Print()
		}
	}

	n.SaveToJson("parameters.json")
}
