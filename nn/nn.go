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

type NeuralNetwork struct {
	config        []configuration
	num_of_imputs int8
	num_of_layers int8
	Inputs        []float32
	layers        [][]neuron
	biases        []float32
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
		tmp = n.biases[lay]
		if lay == 0 {
			for neu := range n.layers[lay] {
				n.layers[lay][neu].value = 0.0
				for in := range n.Inputs {
					tmp += n.Inputs[in] * n.layers[lay][neu].weights[in]
				}
				n.layers[lay][neu].value = activation(tmp)
			}
		} else if lay < int(n.num_of_layers)-1 {
			for neu := range n.layers[lay] {
				tmp = n.biases[lay]
				n.layers[lay][neu].value = 0.0
				for wei := range n.layers[lay][neu].weights {
					tmp += n.layers[lay-1][wei].value * n.layers[lay][neu].weights[wei]
				}
				n.layers[lay][neu].value = activation(tmp)
			}
		} else {
			for neu := range n.layers[lay] {
				n.layers[lay][neu].value = 0.0
				for wei := range n.layers[lay][neu].weights {
					n.layers[lay][neu].value += n.layers[lay-1][wei].value * n.layers[lay][neu].weights[wei]
				}
				n.layers[lay][neu].value += n.biases[lay]
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

		dif := n.layers[n.num_of_layers-1][outIndex].value - n.trainData[row][n.num_of_imputs+outIndex]

		res += dif * dif
	}

	return res / float32(len(n.trainData))
}

func (n *NeuralNetwork) Print() {
	fmt.Println()
	for range n.Inputs {
		fmt.Printf("------------")
	}
	for range n.layers[n.num_of_layers-1] {
		fmt.Printf("------------")
	}
	fmt.Println()

	for ind := range n.Inputs {
		fmt.Printf("|%10s|", "in" + strconv.Itoa(ind))
	}
	for ind := range n.layers[n.num_of_layers-1] {
		fmt.Printf("|%10s|", "out" + strconv.Itoa(ind))
	}
	fmt.Println()

	for ind := range n.Inputs {
		fmt.Printf("|%10.3f|", n.Inputs[ind])
	}
	for ind := range n.layers[n.num_of_layers-1] {
		fmt.Printf("|%10.3f|", n.layers[n.num_of_layers-1][ind].value)
	}
	fmt.Println()
	
	for range n.Inputs {
		fmt.Printf("------------")
	}
	for range n.layers[n.num_of_layers-1] {
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
		return errors.New("количество столбцов данных не соответствует количеству входов и выходов нейросети")
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
				tmp_str := "параметр "
				tmp_str += strconv.Itoa(index + 1)
				tmp_str += " меньше 1"

				return errors.New(tmp_str)
			}
		}
		n.num_of_imputs = int8(c[0])
		n.Inputs = make([]float32, c[0])

		n.num_of_layers = int8(len(c) - 1)
		n.layers = make([][]neuron, n.num_of_layers)
		n.biases = make([]float32, n.num_of_layers)
		for lay := range n.layers {
			n.layers[lay] = make([]neuron, c[lay+1])
			for ws := range n.layers[lay] {
				n.layers[lay][ws].weights = make([]float32, c[lay])
				for wei := range n.layers[lay][ws].weights {
					n.layers[lay][ws].weights[wei] = 0.5
				}
			}
		}

		return nil
	}

	return errors.New("в конфигурации меньше двух элементов")
}

func (n *NeuralNetwork) Train() {
	for out := range n.layers[n.num_of_layers-1] {
		curCost := n.cost(int8(out))

		for lay := range n.layers {
			n.biases[lay] += n.Offset
			newCost := n.cost(int8(out))
			n.biases[lay] -= n.Offset
			n.biases[lay] -= n.Step * ((newCost - curCost) / n.Offset)

			for neu := range n.layers[lay] {
				for wei := range n.layers[lay][neu].weights {
					n.layers[lay][neu].weights[wei] += n.Offset
					newCost = n.cost(int8(out))
					n.layers[lay][neu].weights[wei] -= n.Offset
					n.layers[lay][neu].weights[wei] -= n.Step * ((newCost - curCost) / n.Offset)
				}
			}
		}
	}
}
