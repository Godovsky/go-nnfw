package nn

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
)

type neuron struct {
	Weights []float32 `json:"Weights"`
	Value   float32   `json:"Value"`
	Bias    float32   `json:"Bias"`
}

type layer struct {
	Neurons []neuron `json:"Neurons"`
}

type NeuralNetwork struct {
	Config      []uint8     `json:"Config"`
	Offset      float32     `json:"Offset"`
	Step        float32     `json:"Step"`
	NumOfImputs int8        `json:"Num_of_imputs"`
	NumOfLayers int8        `json:"NumOfLayers"`
	Inputs      []float32   `json:"Inputs"`
	Layers      []layer     `json:"Layers"`
	TrainData   [][]float32 `json:"-"`
}

func activation(n float32) float32 {
	return float32(1.0) / float32(1.0+math.Exp(float64(-n)))
}

func (n *NeuralNetwork) Calculate() {
	var tmp float32
	for lay := range n.Layers {
		if lay == 0 {
			for neu := range n.Layers[lay].Neurons {
				tmp = n.Layers[lay].Neurons[neu].Bias
				for in := range n.Inputs {
					tmp += n.Inputs[in] * n.Layers[lay].Neurons[neu].Weights[in]
				}
				n.Layers[lay].Neurons[neu].Value = activation(tmp)
			}
		} else if lay < int(n.NumOfLayers)-1 {
			for neu := range n.Layers[lay].Neurons {
				tmp = n.Layers[lay].Neurons[neu].Bias
				for wei := range n.Layers[lay].Neurons[neu].Weights {
					tmp += n.Layers[lay-1].Neurons[wei].Value * n.Layers[lay].Neurons[neu].Weights[wei]
				}
				n.Layers[lay].Neurons[neu].Value = activation(tmp)
			}
		} else {
			for neu := range n.Layers[lay].Neurons {
				tmp = /* 0.0 */ n.Layers[lay].Neurons[neu].Bias
				for wei := range n.Layers[lay].Neurons[neu].Weights {
					tmp += n.Layers[lay-1].Neurons[wei].Value * n.Layers[lay].Neurons[neu].Weights[wei]
				}
				n.Layers[lay].Neurons[neu].Value = tmp
			}
		}
	}
}

func (n *NeuralNetwork) cost(outIndex int8) float32 {
	var res float32
	for row := range n.TrainData {
		for in := range n.Inputs {
			n.Inputs[in] = n.TrainData[row][in]
		}
		n.Calculate()

		dif := n.Layers[n.NumOfLayers-1].Neurons[outIndex].Value - n.TrainData[row][n.NumOfImputs+outIndex]

		res += dif * dif
	}

	return res / float32(len(n.TrainData))
}

func (n *NeuralNetwork) Print() {
	fmt.Println()
	for range n.Inputs {
		fmt.Printf("------------")
	}
	for range n.Layers[n.NumOfLayers-1].Neurons {
		fmt.Printf("------------")
	}
	fmt.Println()

	for inp := range n.Inputs {
		fmt.Printf("|%10s|", "in"+strconv.Itoa(inp))
	}
	for neu := range n.Layers[n.NumOfLayers-1].Neurons {
		fmt.Printf("|%10s|", "out"+strconv.Itoa(neu))
	}
	fmt.Println()

	for inp := range n.Inputs {
		fmt.Printf("|%10.5f|", n.Inputs[inp])
	}
	for neu := range n.Layers[n.NumOfLayers-1].Neurons {
		fmt.Printf("|%10.5f|", n.Layers[n.NumOfLayers-1].Neurons[neu].Value)
	}
	fmt.Println()

	for range n.Inputs {
		fmt.Printf("------------")
	}
	for range n.Layers[n.NumOfLayers-1].Neurons {
		fmt.Printf("------------")
	}
}

func (n *NeuralNetwork) GetDataFromFile(data_path string, numOfHeadLines uint) error {
	if n.Config == nil {
		return errors.New("the neural network configuration is emty")
	}

	fd, err := os.OpenFile(data_path, os.O_RDONLY, 0666)
	if err != nil {
		return err
	}

	csvR := csv.NewReader(fd)
	csvR.Comma = ';'
	data_strs, err := csvR.ReadAll()
	fd.Close()
	if err != nil {
		return err
	}

	if len(data_strs[0]) != int(n.Config[0])+int(n.Config[len(n.Config)-1]) {
		return errors.New("the number of data columns does not correspond to the number of inputs and outputs of the neural network")
	}

	n.TrainData = make([][]float32, len(data_strs)-int(numOfHeadLines))
	for ind1 := range data_strs {
		if ind1 == 0 {
			continue
		}

		n.TrainData[ind1-int(numOfHeadLines)] = make([]float32, len(data_strs[ind1]))
		for ind2 := range data_strs[ind1] {
			tmp, err := strconv.ParseFloat(data_strs[ind1][ind2], 32)
			if err != nil {
				return err
			}
			n.TrainData[ind1-int(numOfHeadLines)][ind2] = float32(tmp)
		}
	}
	return err
}

func (n *NeuralNetwork) ClearData() {
	n.TrainData = nil
}

func (n *NeuralNetwork) AddDataLine(data ...float32) error {
	if n.Config == nil {
		return errors.New("the neural network configuration is emty")
	}

	if len(data) != int(n.Config[0])+int(n.Config[len(n.Config)-1]) {
		return errors.New("the number of data columns does not correspond to the number of inputs and outputs of the neural network")
	}

	n.TrainData = append(n.TrainData, data)

	return nil
}

/*
An arguments:

  - offset
  - learning step
  - number of inputs;
  - number of neurons (layer 1);
  - ... ;
  - number of neurons (layer N);
  - number of outputs.

An example with two inputs, two layers of 5 and 4 neurons each, and three outputs:

	err := SetConfiguration(2, 5, 4, 3)
	if err != nil {
		log.Fatal(err)
	}
*/
func (n *NeuralNetwork) Init(offset float32, step float32, config []uint8) error {
	if offset == 0 || step == 0 {
		return errors.New("offset and step can't be less than or equal to zero")
	}
	n.Offset = offset
	n.Step = step

	n.Config = config

	if len(config) >= 2 {
		for index, Value := range config {
			if Value < 1 {
				tmp_str := "the parameter "
				tmp_str += strconv.Itoa(index + 1)
				tmp_str += " is less than 1"

				return errors.New(tmp_str)
			}
		}
		n.NumOfImputs = int8(config[0])
		n.Inputs = make([]float32, config[0])

		n.NumOfLayers = int8(len(config) - 1)
		n.Layers = make([]layer, n.NumOfLayers)
		for lay := range n.Layers {
			n.Layers[lay].Neurons = make([]neuron, config[lay+1])
			for neu := range n.Layers[lay].Neurons {
				n.Layers[lay].Neurons[neu].Weights = make([]float32, config[lay])
				for wei := range n.Layers[lay].Neurons[neu].Weights {
					n.Layers[lay].Neurons[neu].Weights[wei] = 0.0
				}
				n.Layers[lay].Neurons[neu].Bias = 0.0
			}
		}

		return nil
	}

	return errors.New("there are fewer than two elements in the Configuration")
}

func (n *NeuralNetwork) ResetOffsetAndStep(offset float32, step float32) error {
	if offset <= 0 || step <= 0 {
		return errors.New("offset and step can't be less than or equal to zero")
	}
	n.Offset = offset
	n.Step = step

	return nil
}

func (n *NeuralNetwork) Train() {
	for out := range n.Layers[n.NumOfLayers-1].Neurons {
		var curCost float32
		var newCost float32

		curCost = n.cost(int8(out))

		for lay := range n.Layers {
			for neu := range n.Layers[lay].Neurons {
				// if lay != int(n.NumOfLayers-1) {
				origBias := n.Layers[lay].Neurons[neu].Bias
				n.Layers[lay].Neurons[neu].Bias += n.Offset
				newCost = n.cost(int8(out))
				n.Layers[lay].Neurons[neu].Bias = origBias

				dif := newCost - curCost
				gradient := dif / n.Offset

				n.Layers[lay].Neurons[neu].Bias -= n.Step * gradient
				// }

				for wei := range n.Layers[lay].Neurons[neu].Weights {
					origWeight := n.Layers[lay].Neurons[neu].Weights[wei]
					n.Layers[lay].Neurons[neu].Weights[wei] += n.Offset
					newCost = n.cost(int8(out))
					n.Layers[lay].Neurons[neu].Weights[wei] = origWeight

					dif := newCost - curCost
					gradient := dif / n.Offset
					n.Layers[lay].Neurons[neu].Weights[wei] -= n.Step * gradient
				}
			}
		}
	}
}

func (n *NeuralNetwork) SaveToJson(fileName string) error {
	dataJson, err := json.MarshalIndent(n, "", "\t")
	if err != nil {
		return err
	}
	os.WriteFile(fileName, []byte(dataJson), 0777)

	return nil
}

func (n *NeuralNetwork) LoadFromJson(fileName string) error {
	fd, err := os.ReadFile(fileName)
	if err != nil {
		return err
	}

	err = json.Unmarshal(fd, n)
	if err != nil {
		return err
	}

	return nil
}
