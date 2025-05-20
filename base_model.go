package main

import (
	"math"
	"math/rand"
	"sync"
)

// Model architecture constants
const (
	InputSize  = 308  // Number of input features
	HiddenSize = 64   // Number of hidden units
	OutputSize = 4    // Number of output actions (3 continuous + 1 binary)
)

// BaseModel contains common functionality shared between CPU and GPU models
type BaseModel struct {
	mu           sync.RWMutex
	actorWeights  []float32
	criticWeights []float32
	learningRate  float32
	gamma         float32
	lambda        float32
	totalSteps    int64
	epsilon       float32
	metadataMutex sync.RWMutex
}

// NewBaseModel creates a new base model with initialized weights
func NewBaseModel() *BaseModel {
	baseModel := &BaseModel{
		learningRate: 0.0001,  // Keep learning rate low for stability
		gamma:       0.99,
		lambda:      0.95,
		epsilon:     0.8,  // Increased from 0.4 to 0.8 for maximum exploration
	}

	// Initialize weights with Xavier/Glorot initialization
	// input->hidden1 (InputSize->HiddenSize) + hidden1->hidden2 (HiddenSize->HiddenSize) + hidden2->output (HiddenSize->OutputSize)
	baseModel.actorWeights = make([]float32, InputSize*HiddenSize + HiddenSize*HiddenSize + HiddenSize*OutputSize)
	baseModel.criticWeights = make([]float32, HiddenSize*1)  // hidden2->value (HiddenSize->1)
	
	// Xavier/Glorot initialization with gain=1.0 for better stability
	for i := range baseModel.actorWeights {
		if i < InputSize*HiddenSize {  // input->hidden1
			baseModel.actorWeights[i] = float32(rand.NormFloat64()) * float32(math.Sqrt(2.0/float64(InputSize)))  // gain=1.0
		} else if i < InputSize*HiddenSize + HiddenSize*HiddenSize {  // hidden1->hidden2
			baseModel.actorWeights[i] = float32(rand.NormFloat64()) * float32(math.Sqrt(2.0/float64(HiddenSize)))
		} else {  // hidden2->output
			baseModel.actorWeights[i] = float32(rand.NormFloat64()) * float32(math.Sqrt(2.0/float64(HiddenSize)))
		}
	}
	
	for i := range baseModel.criticWeights {
		baseModel.criticWeights[i] = float32(rand.NormFloat64()) * float32(math.Sqrt(2.0/float64(HiddenSize)))
	}

	return baseModel
}

// Helper functions for statistics
func mean(values []float32) float32 {
	var sum float32
	for _, v := range values {
		sum += v
	}
	return sum / float32(len(values))
}

func max(values []float32) float32 {
	var max float32
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max
}

