package nn

import (
	"encoding/csv"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
)

type configuration int8

type neuron struct {
	weights []float32
	value   float32
}

type layer struct {
	neurons []neuron
	bias    float32
}

type NeuralNetwork struct {
	config        []configuration
	num_of_imputs int8
	num_of_layers int8
	Inputs        []float32
	layers        []layer
	trainData     [][]float32
	Offset        float32
	Step          float32
}

func activation(n float32) float32 {
	return float32(1.0) / float32(1.0+math.Exp(float64(-n)))
}

func (n *NeuralNetwork) Calculate() {
	for lay := range n.layers {
		var tmp float32
		tmp = n.layers[lay].bias
		if lay == 0 {
			for neu := range n.layers[lay].neurons {
				n.layers[lay].neurons[neu].value = 0.0
				for in := range n.Inputs {
					tmp += n.Inputs[in] * n.layers[lay].neurons[neu].weights[in]
				}
				n.layers[lay].neurons[neu].value = activation(tmp)
			}
		} else if lay < int(n.num_of_layers)-1 {
			for neu := range n.layers[lay].neurons {
				tmp = n.layers[lay].bias
				n.layers[lay].neurons[neu].value = 0.0
				for wei := range n.layers[lay].neurons[neu].weights {
					tmp += n.layers[lay-1].neurons[wei].value * n.layers[lay].neurons[neu].weights[wei]
				}
				n.layers[lay].neurons[neu].value = activation(tmp)
			}
		} else {
			for neu := range n.layers[lay].neurons {
				n.layers[lay].neurons[neu].value = 0.0
				for wei := range n.layers[lay].neurons[neu].weights {
					n.layers[lay].neurons[neu].value += n.layers[lay-1].neurons[wei].value * n.layers[lay].neurons[neu].weights[wei]
				}
				n.layers[lay].neurons[neu].value += n.layers[lay].bias
			}
		}
	}
}

func (n *NeuralNetwork) cost(outIndex int8) float32 {
	var res float32
	for row := range n.trainData {
		for in := range n.Inputs {
			n.Inputs[in] = n.trainData[row][in]
		}
		n.Calculate()

		dif := n.layers[n.num_of_layers-1].neurons[outIndex].value - n.trainData[row][n.num_of_imputs+outIndex]

		res += dif * dif
	}

	return res / float32(len(n.trainData))
}

func (n *NeuralNetwork) Print() {
	fmt.Println()
	for range n.Inputs {
		fmt.Printf("------------")
	}
	for range n.layers[n.num_of_layers-1].neurons {
		fmt.Printf("------------")
	}
	fmt.Println()

	for inp := range n.Inputs {
		fmt.Printf("|%10s|", "in" + strconv.Itoa(inp))
	}
	for neu := range n.layers[n.num_of_layers-1].neurons {
		fmt.Printf("|%10s|", "out" + strconv.Itoa(neu))
	}
	fmt.Println()

	for inp := range n.Inputs {
		fmt.Printf("|%10.3f|", n.Inputs[inp])
	}
	for neu := range n.layers[n.num_of_layers-1].neurons {
		fmt.Printf("|%10.3f|", n.layers[n.num_of_layers-1].neurons[neu].value)
	}
	fmt.Println()
	
	for range n.Inputs {
		fmt.Printf("------------")
	}
	for range n.layers[n.num_of_layers-1].neurons {
		fmt.Printf("------------")
	}
}

func (n *NeuralNetwork) Constructor(data_path string, c ...configuration) error {
	n.config = c
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

	if len(data_strs[0]) != int(n.config[0])+int(n.config[len(n.config)-1]) {
		return errors.New("the number of data columns does not correspond to the number of inputs and outputs of the neural network")
	}

	n.trainData = make([][]float32, len(data_strs))
	for ind1 := range data_strs {
		n.trainData[ind1] = make([]float32, len(data_strs[ind1]))
		for ind2 := range data_strs[ind1] {
			tmp, err := strconv.ParseFloat(data_strs[ind1][ind2], 32)
			if err != nil {
				return err
			}
			n.trainData[ind1][ind2] = float32(tmp)
		}
	}

	if len(c) >= 2 {
		for index, value := range c {
			if value < 1 {
				tmp_str := "the parameter "
				tmp_str += strconv.Itoa(index + 1)
				tmp_str += " is less than 1"

				return errors.New(tmp_str)
			}
		}
		n.num_of_imputs = int8(c[0])
		n.Inputs = make([]float32, c[0])

		n.num_of_layers = int8(len(c) - 1)
		n.layers = make([]layer, n.num_of_layers)
		for lay := range n.layers {
			n.layers[lay].neurons = make([]neuron, c[lay+1])
			for neu := range n.layers[lay].neurons {
				n.layers[lay].neurons[neu].weights = make([]float32, c[lay])
				for wei := range n.layers[lay].neurons[neu].weights {
					n.layers[lay].neurons[neu].weights[wei] = 0.5
				}
			}
		}

		return nil
	}

	return errors.New("there are fewer than two elements in the configuration")
}

func (n *NeuralNetwork) Train() {
	for out := range n.layers[n.num_of_layers-1].neurons {
		curCost := n.cost(int8(out))

		for lay := range n.layers {
			n.layers[lay].bias += n.Offset
			newCost := n.cost(int8(out))
			n.layers[lay].bias -= n.Offset
			n.layers[lay].bias -= n.Step * ((newCost - curCost) / n.Offset)

			for neu := range n.layers[lay].neurons {
				for wei := range n.layers[lay].neurons[neu].weights {
					n.layers[lay].neurons[neu].weights[wei] += n.Offset
					newCost = n.cost(int8(out))
					n.layers[lay].neurons[neu].weights[wei] -= n.Offset
					n.layers[lay].neurons[neu].weights[wei] -= n.Step * ((newCost - curCost) / n.Offset)
				}
			}
		}
	}
}
