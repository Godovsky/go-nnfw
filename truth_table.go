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

		// config := []uint8{2, 3, 6}
		// err = n.SetConfiguration(0.01, 0.01, config)
		err = n.Init(0.01, 0.01, []uint8{2, 3, 6})
		if err != nil {
			log.Fatalln(err)
		}
	}

	// n.ResetEpsilonAndStep(0.01, 0.001)

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

	for range 50 {
		for range 10000 {
			n.Train()
		}

		for i := range 2 {
			for j := range 2 {
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
