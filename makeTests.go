package main

import (
	"fmt"
	"math/rand"
	"time"

	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)

func main() {
	/* first example */
	// appendInSimpleLoop()

	/* second example */
	rand.Seed(1)
	fmt.Println("init slice capacity with zero:")
	appendInBasicLoop("zero")
	fmt.Println("init slice capacity with calculated final length:")
	appendInBasicLoop("calc")
	fmt.Println("init slice capacity with calculated final length from function:")
	data := appendInBasicLoop("func")
	// fmt.Println(data)
	fmt.Println("train ML model:")
	trainModel(data)
	fmt.Println("init slice capacity with model's prediction:")
	appendInBasicLoop("model")
	fmt.Println("init slice capacity with model's prediction+1:")
	appendInBasicLoop("model+1")
	rand.Seed(time.Now().Unix())
	fmt.Println("init slice capacity with model's prediction and use new seed for random input: ")
	appendInBasicLoop("model")
}

func appendInSimpleLoop() {
	times := 1000
	results := make([]time.Duration, 0, times)
	for t := 0; t < times; t++ {
		start := time.Now()
		// initialize either with capacity of 0
		// test := make([]int, 0)
		// or initialize with final capacity of 100
		test := make([]int, 0, 100)
		for i := 0; i < 100; i++ {
			// uncomment to see how the capacity grows in larger steps
			// fmt.Println(cap(test), len(test))
			test = append(test, i)
		}
		elapsed := time.Now().Sub(start)
		// fmt.Println(elapsed.Nanoseconds())
		results = append(results, elapsed)
	}
	printSummary(results)
}

/* printSummary prints out average, maximum, minimum and sum in nanoseconds for a slice of durations */
func printSummary(results []time.Duration) {
	max := results[0]
	min := results[0]
	var sum time.Duration
	for _, r := range results {
		if r < min {
			min = r
		}
		if r > max {
			max = r
		}
		sum += r
	}
	avg := int(sum) / len(results)
	fmt.Println("avg: ", avg, "max:", max.Nanoseconds(), "min: ", min.Nanoseconds(), "sum: ", sum.Nanoseconds())
}

/* second example */

var net *deep.Neural

func init() {
	net = deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{2, 5, 1},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeRegression,
		Weight:     deep.NewUniform(0.1, 0.0),
		Bias:       true,
	})
}

func getSamples(size int) []int {
	samp := make([]int, size)
	for i := 0; i < size; i++ {
		samp[i] = rand.Intn(500)
	}
	return samp
}

func appendInBasicLoop(kind string) training.Examples {
	samp := getSamples(1000)
	data := make([]training.Example, 0, len(samp))
	times := 1000
	results := make([]time.Duration, 0, times)
	for trys := 0; trys < times; trys++ {
		start := time.Now()
		for j, s := range samp {
			var test []int
			switch kind {
			case "zero":
				test = make([]int, 0)
			case "calc":
				test = make([]int, 0, s*3+j)
			case "func":
				test = make([]int, 0, getCap(s, j))
			case "model":
				// uncomment to check out how close the predictions are to actual values
				// c := getCapFromModel(s, j)
				// if c != s*3+j {
				// 	fmt.Println("model: ", c, " actual: ", s*3+j)
				// }
				test = make([]int, 0, getCapFromModel(s, j))
			case "model+1":
				test = make([]int, 0, getCapFromModel(s, j)+1)
			}
			for i := 0; i < s; i++ {
				// fmt.Println(cap(test), len(test))
				test = append(test, i)
				test = append(test, j)
				test = append(test, s)
			}
			for k := 0; k < j; k++ {
				test = append(test, k)
			}
			data = append(data,
				training.Example{
					Input:    []float64{float64(s), float64(j)},
					Response: []float64{float64(len(test))}})
		}
		elapsed := time.Now().Sub(start)
		// fmt.Println(elapsed.Nanoseconds())
		results = append(results, elapsed)
	}
	printSummary(results)
	return data
}

func getCap(s, j int) int {
	return s*3 + j
}

func getCapFromModel(s, j int) int {
	p := net.Predict([]float64{float64(s), float64(j)})
	return int(p[0])
}

func trainModel(data training.Examples) {
	optimizer := training.NewAdam(0.02, 0.9, 0.999, 1e-8)
	trainer := training.NewBatchTrainer(optimizer, 1, 200, 8)

	training, heldout := data.Split(0.75)
	trainer.Train(net, training, heldout, 7)
}
