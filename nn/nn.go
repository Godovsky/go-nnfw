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

type Configuration int8

type neuron struct {
	Weights []float32 `json:"Weights"`
	Value   float32   `json:"Value"`
}

type layer struct {
	Neurons []neuron `json:"Neurons"`
	Bias    float32  `json:"Bias"`
}

type NeuralNetwork struct {
	Config      []Configuration `json:"Config"`
	Offset      float32         `json:"Offset"`
	Step        float32         `json:"Step"`
	NumOfImputs int8            `json:"Num_of_imputs"`
	NumOfLayers int8            `json:"NumOfLayers"`
	Inputs      []float32       `json:"Inputs"`
	Layers      []layer         `json:"Layers"`
	TrainData   [][]float32     `json:"-"`
}

func activation(n float32) float32 {
	return float32(1.0) / float32(1.0+math.Exp(float64(-n)))
}

func (n *NeuralNetwork) Calculate() {
	for lay := range n.Layers {
		var tmp float32
		tmp = n.Layers[lay].Bias
		if lay == 0 {
			for neu := range n.Layers[lay].Neurons {
				n.Layers[lay].Neurons[neu].Value = 0.0
				for in := range n.Inputs {
					tmp += n.Inputs[in] * n.Layers[lay].Neurons[neu].Weights[in]
				}
				n.Layers[lay].Neurons[neu].Value = activation(tmp)
			}
		} else if lay < int(n.NumOfLayers)-1 {
			for neu := range n.Layers[lay].Neurons {
				tmp = n.Layers[lay].Bias
				n.Layers[lay].Neurons[neu].Value = 0.0
				for wei := range n.Layers[lay].Neurons[neu].Weights {
					tmp += n.Layers[lay-1].Neurons[wei].Value * n.Layers[lay].Neurons[neu].Weights[wei]
				}
				n.Layers[lay].Neurons[neu].Value = activation(tmp)
			}
		} else {
			for neu := range n.Layers[lay].Neurons {
				n.Layers[lay].Neurons[neu].Value = 0.0
				for wei := range n.Layers[lay].Neurons[neu].Weights {
					n.Layers[lay].Neurons[neu].Value += n.Layers[lay-1].Neurons[wei].Value * n.Layers[lay].Neurons[neu].Weights[wei]
				}
				// n.Layers[lay].Neurons[neu].Value += n.Layers[lay].Bias
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
		fmt.Printf("|%10.3f|", n.Inputs[inp])
	}
	for neu := range n.Layers[n.NumOfLayers-1].Neurons {
		fmt.Printf("|%10.3f|", n.Layers[n.NumOfLayers-1].Neurons[neu].Value)
	}
	fmt.Println()

	for range n.Inputs {
		fmt.Printf("------------")
	}
	for range n.Layers[n.NumOfLayers-1].Neurons {
		fmt.Printf("------------")
	}
}

func (n *NeuralNetwork) GetDataFromFile(data_path string) error {
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

	n.TrainData = make([][]float32, len(data_strs))
	for ind1 := range data_strs {
		n.TrainData[ind1] = make([]float32, len(data_strs[ind1]))
		for ind2 := range data_strs[ind1] {
			tmp, err := strconv.ParseFloat(data_strs[ind1][ind2], 32)
			if err != nil {
				return err
			}
			n.TrainData[ind1][ind2] = float32(tmp)
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

func (n *NeuralNetwork) Constructor(c ...Configuration) error {
	n.Config = c

	if len(c) >= 2 {
		for index, Value := range c {
			if Value < 1 {
				tmp_str := "the parameter "
				tmp_str += strconv.Itoa(index + 1)
				tmp_str += " is less than 1"

				return errors.New(tmp_str)
			}
		}
		n.NumOfImputs = int8(c[0])
		n.Inputs = make([]float32, c[0])

		n.NumOfLayers = int8(len(c) - 1)
		n.Layers = make([]layer, n.NumOfLayers)
		for lay := range n.Layers {
			n.Layers[lay].Neurons = make([]neuron, c[lay+1])
			for neu := range n.Layers[lay].Neurons {
				n.Layers[lay].Neurons[neu].Weights = make([]float32, c[lay])
				for wei := range n.Layers[lay].Neurons[neu].Weights {
					n.Layers[lay].Neurons[neu].Weights[wei] = 0.5
				}
			}
		}

		return nil
	}

	return errors.New("there are fewer than two elements in the Configuration")
}

func (n *NeuralNetwork) Train() {
	for out := range n.Layers[n.NumOfLayers-1].Neurons {
		curCost := n.cost(int8(out))

		for lay := range n.Layers {
			n.Layers[lay].Bias += n.Offset
			newCost := n.cost(int8(out))
			n.Layers[lay].Bias -= n.Offset
			n.Layers[lay].Bias -= n.Step * ((newCost - curCost) / n.Offset)

			for neu := range n.Layers[lay].Neurons {
				for wei := range n.Layers[lay].Neurons[neu].Weights {
					n.Layers[lay].Neurons[neu].Weights[wei] += n.Offset
					newCost = n.cost(int8(out))
					n.Layers[lay].Neurons[neu].Weights[wei] -= n.Offset
					n.Layers[lay].Neurons[neu].Weights[wei] -= n.Step * ((newCost - curCost) / n.Offset)
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
