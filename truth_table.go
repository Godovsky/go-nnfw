package main

import (
	"log"
	"nn/nn"
)

func main() {
	var n nn.NeuralNetwork

	err := n.LoadFromJson("parameters.json")
	if err != nil {
		log.Println(err)

		// config := []uint8{2, 6, 6}
		// err = n.SetConfiguration(0.01, 0.001, config)
		err = n.Init(0.01, 0.001, []uint8{2, 4, 6})
		if err != nil {
			log.Fatalln(err)
		}
	}

	// n.ResetOffsetAndStep(0.01, 0.001)

	err = n.GetDataFromFile("data.csv", 1)
	if err != nil {
		log.Fatalln(err)
	}

	// You can add the data line
	// err = n.AddDataLine(1, 1, 1, 1, 0, 0, 0, 1)
	// if err != nil {
	// 	fmt.Println(err)
	// 	return
	// }
	// fmt.Println(n.TrainData)

	for i := 0; i < 50; i++ {
		for j := 0; j < 10000; j++ {
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
	}

	err = n.SaveToJson("parameters.json")
	if err != nil {
		log.Println(err)
	}
}
