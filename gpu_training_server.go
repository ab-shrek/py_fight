package main

/*
#cgo LDFLAGS: -L/usr/local/cuda-12.2/lib64 -lcuda -lcudart
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

// CUDA device attributes
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
*/
import "C"
import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/gin-gonic/gin"
	"github.com/shirou/gopsutil/v3/process"
)

// CUDA kernel definitions
const cudaKernels = `
// CUDA kernels are now defined in kernels.cu
`

// result converts a CUDA result to an error
func result(r C.CUresult) error {
	if r == C.CUDA_SUCCESS {
		return nil
	}
	var str *C.char
	C.cuGetErrorString(r, &str)
	return fmt.Errorf("CUDA error: %s", C.GoString(str))
}

// Constants
const (
	NumWorkers     = 40
	BatchSize      = 1024
	BufferSize     = 1000000
	MinBufferSize  = 10000
	MaxConnections = 1000
	CheckpointDir  = "checkpoints"
	LogDir         = "logs"
	UseGPU         = true  // Enable GPU by default
	Port           = 5000
	CUDATimeout    = 30 * time.Second  // Increased timeout for CUDA operations
	MaxRetries     = 3     // Maximum number of retries for CUDA operations
)

// Global variables for CUDA
var (
	contextMutex sync.Mutex
	globalContext C.CUcontext
	contextInitialized bool
)

// Initialize CUDA and create context
func initializeCUDA() (*C.CUcontext, error) {
	contextMutex.Lock()
	defer contextMutex.Unlock()

	if contextInitialized {
		return &globalContext, nil
	}

	log.Println("Initializing CUDA...")
	
	// Initialize CUDA driver with retries
	var initErr error
	for i := 0; i < MaxRetries; i++ {
		if err := result(C.cuInit(0)); err != nil {
			initErr = err
			time.Sleep(time.Second) // Wait before retry
			continue
		}
		initErr = nil
		break
	}
	if initErr != nil {
		return nil, fmt.Errorf("failed to initialize CUDA driver after %d retries: %v", MaxRetries, initErr)
	}
	log.Println("CUDA driver initialized successfully")
	
	// Get device count
	var deviceCount C.int
	if err := result(C.cuDeviceGetCount(&deviceCount)); err != nil {
		return nil, fmt.Errorf("failed to get device count: %v", err)
	}
	
	if deviceCount == 0 {
		return nil, fmt.Errorf("no CUDA devices found")
	}
	
	log.Printf("Found %d CUDA devices", deviceCount)
	
	// Get best device
	var bestDevice C.CUdevice
	var maxMemory int64
	for i := 0; i < int(deviceCount); i++ {
		var device C.CUdevice
		if err := result(C.cuDeviceGet(&device, C.int(i))); err != nil {
			continue
		}
		
		var name [256]C.char
		if err := result(C.cuDeviceGetName(&name[0], 256, device)); err != nil {
			continue
		}
		
		var computeCap [2]C.int
		if err := result(C.cuDeviceGetAttribute(&computeCap[0], C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)); err != nil {
			continue
		}
		if err := result(C.cuDeviceGetAttribute(&computeCap[1], C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)); err != nil {
			continue
		}
		
		var totalMem C.size_t
		if err := result(C.cuDeviceTotalMem(&totalMem, device)); err != nil {
			continue
		}
		
		log.Printf("Device %d: %s (Compute Capability: %d.%d, Memory: %d MB)",
			i, C.GoString(&name[0]), computeCap[0], computeCap[1], totalMem/1024/1024)
		
		if int64(totalMem) > maxMemory {
			maxMemory = int64(totalMem)
			bestDevice = device
		}
	}
	
	if bestDevice == 0 {
		return nil, fmt.Errorf("no suitable CUDA device found")
	}
	
	log.Printf("Selected device %d as best device", bestDevice)
	
	// Create context with flags for better compatibility
	var ctx C.CUcontext
	log.Println("Creating primary CUDA context...")
	flags := C.uint(C.CU_CTX_SCHED_AUTO) // Use automatic scheduling
	if err := result(C.cuCtxCreate(&ctx, flags, bestDevice)); err != nil {
		return nil, fmt.Errorf("failed to create primary context: %v", err)
	}
	log.Println("Primary CUDA context created successfully")
	
	// Set as current context with timeout and retries
	var setCurrentErr error
	for i := 0; i < MaxRetries; i++ {
		done := make(chan error, 1)
		go func() {
			done <- result(C.cuCtxSetCurrent(ctx))
		}()
		
		select {
		case err := <-done:
			if err != nil {
				setCurrentErr = err
				time.Sleep(time.Second) // Wait before retry
				continue
			}
			setCurrentErr = nil
			break
		case <-time.After(CUDATimeout):
			setCurrentErr = fmt.Errorf("operation timed out after %v", CUDATimeout)
			time.Sleep(time.Second) // Wait before retry
			continue
		}
	}
	
	if setCurrentErr != nil {
		C.cuCtxDestroy(ctx)
		return nil, fmt.Errorf("failed to set primary context as current after %d retries: %v", MaxRetries, setCurrentErr)
	}
	
	log.Println("CUDA context set as current")
	
	// Get device properties
	var name [256]C.char
	if err := result(C.cuDeviceGetName(&name[0], 256, bestDevice)); err != nil {
		C.cuCtxDestroy(ctx)
		return nil, fmt.Errorf("failed to get device name: %v", err)
	}
	
	var computeCap [2]C.int
	if err := result(C.cuDeviceGetAttribute(&computeCap[0], C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, bestDevice)); err != nil {
		C.cuCtxDestroy(ctx)
		return nil, fmt.Errorf("failed to get compute capability: %v", err)
	}
	if err := result(C.cuDeviceGetAttribute(&computeCap[1], C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, bestDevice)); err != nil {
		C.cuCtxDestroy(ctx)
		return nil, fmt.Errorf("failed to get compute capability: %v", err)
	}
	
	var totalMem C.size_t
	if err := result(C.cuDeviceTotalMem(&totalMem, bestDevice)); err != nil {
		C.cuCtxDestroy(ctx)
		return nil, fmt.Errorf("failed to get total memory: %v", err)
	}
	
	log.Printf("Using device: %s (Compute Capability: %d.%d, Memory: %d MB)",
		C.GoString(&name[0]), computeCap[0], computeCap[1], totalMem/1024/1024)
	
	globalContext = ctx
	contextInitialized = true
	return &ctx, nil
}

// GPUConfig holds GPU-related configuration
type GPUConfig struct {
	DeviceID     int
	IsAvailable  bool
	DeviceName   string
	MemoryTotal  int64
	MemoryFree   int64
	ComputeCap   [2]int
}

// Global GPU configuration
var (
	gpuConfig GPUConfig
)

// Add this helper function near the top
func cleanupCUDAContext(ctx C.CUcontext) {
	if ctx != nil {
		// Try to destroy the context, but don't fail if it doesn't work
		C.cuCtxDestroy(ctx)
	}
}

// Modify the makeContextCurrent function
func makeContextCurrent(ctx C.CUcontext) error {
    // First try to get current context without locking
    var currentCtx C.CUcontext
    if err := result(C.cuCtxGetCurrent(&currentCtx)); err != nil {
        log.Printf("Warning: Failed to get current context: %v", err)
    } else if currentCtx == ctx {
        log.Printf("Context already current")
        return nil
    }

    // If we need to switch contexts, do it with a timeout
    done := make(chan error, 1)
    go func() {
        contextMutex.Lock()
        defer contextMutex.Unlock()

        if ctx == nil {
            if !contextInitialized {
                done <- fmt.Errorf("no CUDA context available")
                return
            }
            ctx = globalContext
        }

        // Try to set the context directly first
        err := result(C.cuCtxSetCurrent(ctx))
        if err != nil {
            log.Printf("Warning: Direct context set failed: %v, trying alternative approach", err)
            
            // Try alternative approach: destroy and recreate context
            if currentCtx != nil {
                C.cuCtxDestroy(currentCtx)
            }
            
            // Get the device from the context
            var device C.CUdevice
            if err := result(C.cuCtxGetDevice(&device)); err != nil {
                log.Printf("Warning: Failed to get device from context: %v", err)
                done <- err
                return
            }
            
            // Create a new context
            var newCtx C.CUcontext
            if err := result(C.cuCtxCreate(&newCtx, 0, device)); err != nil {
                log.Printf("Warning: Failed to create new context: %v", err)
                done <- err
                return
            }
            
            // Try to set the new context
            if err := result(C.cuCtxSetCurrent(newCtx)); err != nil {
                log.Printf("Warning: Failed to set new context: %v", err)
                C.cuCtxDestroy(newCtx)
                done <- err
                return
            }
            
            ctx = newCtx
        }
        
        // Verify context was set
        if err := result(C.cuCtxGetCurrent(&currentCtx)); err != nil {
            log.Printf("Warning: Failed to verify context after switch: %v", err)
            done <- err
            return
        }
        
        if currentCtx != ctx {
            log.Printf("Error: Context mismatch! Expected: %v, Got: %v", ctx, currentCtx)
            done <- fmt.Errorf("context mismatch")
            return
        }
        
        globalContext = ctx
        contextInitialized = true
        done <- nil
    }()

    // Wait for context switch with timeout
    select {
    case err := <-done:
        return err
    case <-time.After(CUDATimeout):
        return fmt.Errorf("context switch timed out after %v", CUDATimeout)
    }
}

// withTimeout runs a function with a timeout
func withTimeout(timeout time.Duration, operation func() error) error {
	done := make(chan error, 1)
	go func() {
		done <- operation()
	}()
	
	select {
	case err := <-done:
		return err
	case <-time.After(timeout):
		return fmt.Errorf("operation timed out after %v", timeout)
	}
}

// initGPU initializes CUDA and selects the best available device
func initGPU() (*GPUConfig, error) {
	contextMutex.Lock()
	defer contextMutex.Unlock()

	if !UseGPU {
		log.Printf("GPU usage is disabled, running in CPU mode")
		return &GPUConfig{IsAvailable: false}, nil
	}

	log.Printf("Initializing CUDA...")
	
	// Initialize CUDA driver with retries
	var initErr error
	for i := 0; i < MaxRetries; i++ {
		if err := result(C.cuInit(0)); err != nil {
			initErr = err
			log.Printf("CUDA initialization attempt %d failed: %v", i+1, err)
			time.Sleep(3 * time.Second) // Wait before retry
			continue
		}
		initErr = nil
		break
	}
	if initErr != nil {
		log.Printf("Failed to initialize CUDA driver: %v", initErr)
		return &GPUConfig{IsAvailable: false}, fmt.Errorf("failed to initialize CUDA driver: %v", initErr)
	}
	log.Printf("CUDA driver initialized successfully")

	// Get device count
	var deviceCount C.int
	if err := result(C.cuDeviceGetCount(&deviceCount)); err != nil {
		log.Printf("Failed to get device count: %v", err)
		return &GPUConfig{IsAvailable: false}, fmt.Errorf("failed to get device count: %v", err)
	}

	if deviceCount == 0 {
		log.Printf("No CUDA devices found")
		return &GPUConfig{IsAvailable: false}, fmt.Errorf("no CUDA devices found")
	}

	log.Printf("Found %d CUDA devices", deviceCount)
	
	// Get best device
	var bestDevice C.CUdevice
	var maxMemory int64
	var bestDeviceIndex int = -1
	
	for i := 0; i < int(deviceCount); i++ {
		var device C.CUdevice
		if err := result(C.cuDeviceGet(&device, C.int(i))); err != nil {
			log.Printf("Failed to get device %d: %v", i, err)
			continue
		}
		
		var name [256]C.char
		if err := result(C.cuDeviceGetName(&name[0], 256, device)); err != nil {
			log.Printf("Failed to get name for device %d: %v", i, err)
			continue
		}
		
		var computeCap [2]C.int
		if err := result(C.cuDeviceGetAttribute(&computeCap[0], C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)); err != nil {
			log.Printf("Failed to get compute capability major for device %d: %v", i, err)
			continue
		}
		if err := result(C.cuDeviceGetAttribute(&computeCap[1], C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)); err != nil {
			log.Printf("Failed to get compute capability minor for device %d: %v", i, err)
			continue
		}
		
		var totalMem C.size_t
		if err := result(C.cuDeviceTotalMem(&totalMem, device)); err != nil {
			log.Printf("Failed to get total memory for device %d: %v", i, err)
			continue
		}
		
		log.Printf("Device %d: %s (Compute Capability: %d.%d, Memory: %d MB)",
			i, C.GoString(&name[0]), computeCap[0], computeCap[1], totalMem/1024/1024)
		
		// Check if device is suitable (compute capability >= 3.0)
		if computeCap[0] >= 3 {
			if int64(totalMem) > maxMemory {
				maxMemory = int64(totalMem)
				bestDevice = device
				bestDeviceIndex = i
			}
		} else {
			log.Printf("Device %d skipped: compute capability %d.%d below minimum requirement (3.0)",
				i, computeCap[0], computeCap[1])
		}
	}
	
	if bestDeviceIndex == -1 {
		log.Printf("No suitable CUDA device found (minimum compute capability 3.0 required)")
		return &GPUConfig{IsAvailable: false}, fmt.Errorf("no suitable CUDA device found")
	}
	
	log.Printf("Selected device %d as best device", bestDeviceIndex)
	
	// Create context with flags for better compatibility
	var ctx C.CUcontext
	log.Println("Creating primary CUDA context...")
	flags := C.uint(C.CU_CTX_SCHED_AUTO) // Use automatic scheduling
	if err := result(C.cuCtxCreate(&ctx, flags, bestDevice)); err != nil {
		log.Printf("Failed to create primary context: %v", err)
		return &GPUConfig{IsAvailable: false}, fmt.Errorf("failed to create primary context: %v", err)
	}
	log.Println("Primary CUDA context created successfully")
	
	// Set as current context with timeout and retries
	var setCurrentErr error
	for i := 0; i < MaxRetries; i++ {
		done := make(chan error, 1)
		go func() {
			done <- result(C.cuCtxSetCurrent(ctx))
		}()
		
		select {
		case err := <-done:
			if err != nil {
				setCurrentErr = err
				log.Printf("Failed to set context as current (attempt %d): %v", i+1, err)
				time.Sleep(3 * time.Second) // Wait before retry
				continue
			}
			setCurrentErr = nil
			break
		case <-time.After(CUDATimeout):
			setCurrentErr = fmt.Errorf("operation timed out after %v", CUDATimeout)
			log.Printf("Context set operation timed out (attempt %d)", i+1)
			time.Sleep(3 * time.Second) // Wait before retry
			continue
		}
	}
	
	if setCurrentErr != nil {
		log.Printf("Failed to set primary context as current after %d retries: %v", MaxRetries, setCurrentErr)
		C.cuCtxDestroy(ctx)
		return &GPUConfig{IsAvailable: false}, fmt.Errorf("failed to set primary context as current: %v", setCurrentErr)
	}
	
	log.Println("CUDA context set as current")
	
	// Get device properties for the selected device
	var name [256]C.char
	if err := result(C.cuDeviceGetName(&name[0], 256, bestDevice)); err != nil {
		log.Printf("Failed to get device name: %v", err)
		C.cuCtxDestroy(ctx)
		return &GPUConfig{IsAvailable: false}, fmt.Errorf("failed to get device name: %v", err)
	}
	
	var computeCap [2]C.int
	if err := result(C.cuDeviceGetAttribute(&computeCap[0], C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, bestDevice)); err != nil {
		log.Printf("Failed to get compute capability major: %v", err)
		C.cuCtxDestroy(ctx)
		return &GPUConfig{IsAvailable: false}, fmt.Errorf("failed to get compute capability: %v", err)
	}
	if err := result(C.cuDeviceGetAttribute(&computeCap[1], C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, bestDevice)); err != nil {
		log.Printf("Failed to get compute capability minor: %v", err)
		C.cuCtxDestroy(ctx)
		return &GPUConfig{IsAvailable: false}, fmt.Errorf("failed to get compute capability: %v", err)
	}
	
	var totalMem C.size_t
	if err := result(C.cuDeviceTotalMem(&totalMem, bestDevice)); err != nil {
		log.Printf("Failed to get total memory: %v", err)
		C.cuCtxDestroy(ctx)
		return &GPUConfig{IsAvailable: false}, fmt.Errorf("failed to get total memory: %v", err)
	}
	
	log.Printf("Using device: %s (Compute Capability: %d.%d, Memory: %d MB)",
		C.GoString(&name[0]), computeCap[0], computeCap[1], totalMem/1024/1024)
	
	// Store the context globally
	globalContext = ctx
	contextInitialized = true
	
	return &GPUConfig{
		DeviceID:    bestDeviceIndex,
		IsAvailable: true,
		DeviceName:  C.GoString(&name[0]),
		MemoryTotal: int64(totalMem),
		MemoryFree:  int64(totalMem),
		ComputeCap:  [2]int{int(computeCap[0]), int(computeCap[1])},
	}, nil
}

// Experience represents a single training experience
type Experience struct {
	Observation []float32 `json:"observation"`
	Action      []float32 `json:"action"`
	Reward      float32   `json:"reward"`
	Value       float32   `json:"value"`
	NextValue   float32   `json:"next_value"`
	LogProb     float32   `json:"log_prob"`
	Done        bool      `json:"done"`
}

// TrainingMetadata tracks training statistics
type TrainingMetadata struct {
	TotalSteps    int64     `json:"total_steps"`
	TotalEpisodes int64     `json:"total_episodes"`
	TotalReward   float32   `json:"total_reward"`
	LossHistory   []float32 `json:"loss_history"`
	RewardHistory []float32 `json:"reward_history"`
	LastSave      string    `json:"last_save"`
	MemoryUsage   []MemoryStats `json:"memory_usage"`
}

// MemoryStats tracks memory usage
type MemoryStats struct {
	Timestamp        string  `json:"timestamp"`
	RSS             uint64  `json:"rss"`
	VMS             uint64  `json:"vms"`
	CPUPercent      float64 `json:"cpu_percent"`
	ActiveConnections int   `json:"active_connections"`
	BufferSize      int     `json:"buffer_size"`
	OpenFiles       int     `json:"open_files"`
}

// ModelCheckpoint represents a saved model state
type ModelCheckpoint struct {
	Timestamp     time.Time `json:"timestamp"`
	TotalSteps    int64     `json:"total_steps"`
	TotalEpisodes int64     `json:"total_episodes"`
	TotalReward   float32   `json:"total_reward"`
	LossHistory   []float32 `json:"loss_history"`
	Weights       struct {
		Actor  []float32 `json:"actor"`
		Critic []float32 `json:"critic"`
	} `json:"weights"`
}

// ExperienceBuffer manages the experience replay buffer
type ExperienceBuffer struct {
	data    []Experience
	maxSize int
	mu      sync.RWMutex
	// Priority sampling
	priorities []float32
	alpha      float32 // Priority exponent
	beta       float32 // Importance sampling exponent
	// Enhanced sampling
	tdErrors    []float32 // Temporal difference errors
	episodeEnds []int     // Indices of episode ends
	episodeIdx  int       // Current episode index
}

// NewExperienceBuffer creates a new experience buffer
func NewExperienceBuffer(maxSize int) *ExperienceBuffer {
	return &ExperienceBuffer{
		data:         make([]Experience, 0, maxSize),
		maxSize:      maxSize,
		priorities:   make([]float32, 0, maxSize),
		tdErrors:     make([]float32, 0, maxSize),
		episodeEnds:  make([]int, 0, maxSize/100), // Assuming average episode length of 100
		alpha:        0.6,  // Priority exponent
		beta:         0.4,  // Importance sampling exponent
		episodeIdx:   0,
	}
}

// Add adds an experience to the buffer with priority
func (b *ExperienceBuffer) Add(exp Experience, priority float32) {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	if len(b.data) >= b.maxSize {
		b.data = b.data[1:]
		b.priorities = b.priorities[1:]
		b.tdErrors = b.tdErrors[1:]
		// Update episode ends
		for i := range b.episodeEnds {
			b.episodeEnds[i]--
		}
		if len(b.episodeEnds) > 0 && b.episodeEnds[0] < 0 {
			b.episodeEnds = b.episodeEnds[1:]
		}
	}
	
	b.data = append(b.data, exp)
	b.priorities = append(b.priorities, priority)
	b.tdErrors = append(b.tdErrors, 0)
	
	if exp.Done {
		b.episodeEnds = append(b.episodeEnds, len(b.data)-1)
		b.episodeIdx++
	}
}

// GetBatch returns a batch of experiences using enhanced sampling
func (b *ExperienceBuffer) GetBatch(size int) ([]Experience, []float32) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	
	if len(b.data) < size {
		return nil, nil
	}
	
	// Compute sampling probabilities using multiple strategies
	probs := make([]float32, len(b.priorities))
	
	// 1. Priority-based sampling
	for i, p := range b.priorities {
		probs[i] = float32(math.Pow(float64(p), float64(b.alpha)))
	}
	
	// 2. TD-error based sampling
	for i, td := range b.tdErrors {
		probs[i] += float32(math.Abs(float64(td)))
	}
	
	// 3. Episode-based sampling (favor recent episodes)
	if len(b.episodeEnds) > 0 {
		episodeWeights := make([]float32, len(b.episodeEnds))
		for i := range episodeWeights {
			episodeWeights[i] = float32(math.Exp(float64(i) * 0.1)) // Exponential decay
		}
		
		// Normalize episode weights
		sum := float32(0)
		for _, w := range episodeWeights {
			sum += w
		}
		for i := range episodeWeights {
			episodeWeights[i] /= sum
		}
		
		// Apply episode weights to experiences
		episodeIdx := 0
		for i := range probs {
			if episodeIdx < len(b.episodeEnds) && i > b.episodeEnds[episodeIdx] {
				episodeIdx++
			}
			if episodeIdx < len(episodeWeights) {
				probs[i] *= episodeWeights[episodeIdx]
			}
		}
	}
	
	// Normalize probabilities
	sum := float32(0)
	for _, p := range probs {
		sum += p
	}
	for i := range probs {
		probs[i] /= sum
	}
	
	// Sample indices
	indices := make([]int, size)
	weights := make([]float32, size)
	for i := 0; i < size; i++ {
		// Sample index based on probabilities
		r := rand.Float32()
		sum := float32(0)
		for j, p := range probs {
			sum += p
			if r <= sum {
				indices[i] = j
				break
			}
		}
		
		// Compute importance sampling weight
		weights[i] = float32(math.Pow(float64(float32(len(b.data))*probs[indices[i]]), float64(-b.beta)))
	}
	
	// Get experiences and normalize weights
	experiences := make([]Experience, size)
	maxWeight := float32(0)
	for i, idx := range indices {
		experiences[i] = b.data[idx]
		if weights[i] > maxWeight {
			maxWeight = weights[i]
		}
	}
	
	// Normalize weights
	for i := range weights {
		weights[i] /= maxWeight
	}
	
	return experiences, weights
}

// UpdateTDErrors updates temporal difference errors
func (b *ExperienceBuffer) UpdateTDErrors(tdErrors []float32) {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	for i, td := range tdErrors {
		if i < len(b.tdErrors) {
			b.tdErrors[i] = td
		}
	}
}

// LearningRateScheduler manages learning rate adjustments
type LearningRateScheduler struct {
	mu sync.RWMutex
	initialLR float32
	currentLR float32
	factor    float32
	patience  int
	bestValue float32
	waitCount int
	mode      string // "max" or "min"
}

// NewLearningRateScheduler creates a new learning rate scheduler
func NewLearningRateScheduler(initialLR float32, factor float32, patience int, mode string) *LearningRateScheduler {
	return &LearningRateScheduler{
		initialLR: initialLR,
		currentLR: initialLR,
		factor:    factor,
		patience:  patience,
		bestValue: 0,
		waitCount: 0,
		mode:      mode,
	}
}

// Step updates the learning rate based on the metric
func (s *LearningRateScheduler) Step(metric float32) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.mode == "max" {
		if metric > s.bestValue {
			s.bestValue = metric
			s.waitCount = 0
		} else {
			s.waitCount++
		}
	} else {
		if metric < s.bestValue || s.bestValue == 0 {
			s.bestValue = metric
			s.waitCount = 0
		} else {
			s.waitCount++
		}
	}

	if s.waitCount >= s.patience {
		s.currentLR *= s.factor
		s.waitCount = 0
		log.Printf("Learning rate adjusted to: %f", s.currentLR)
	}
}

// GetLR returns the current learning rate
func (s *LearningRateScheduler) GetLR() float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.currentLR
}

// Model represents the neural network model
type Model struct {
	mu sync.RWMutex
	// Model parameters
	actorWeights  []float32
	criticWeights []float32
	// GPU buffers
	gpuBuffers struct {
		actorWeights  C.CUdeviceptr
		criticWeights C.CUdeviceptr
		// Pre-allocated buffers for forward pass
		input        C.CUdeviceptr
		hidden       C.CUdeviceptr
		hidden2      C.CUdeviceptr  // Added hidden2 buffer
		output       C.CUdeviceptr
		value        C.CUdeviceptr
	}
	// CUDA context
	cudaContext C.CUcontext
	// CUDA streams
	cudaStreams struct {
		 compute C.CUstream
		 memory  C.CUstream
	}
	// CUDA kernels
	kernels struct {
		forwardHidden  C.CUfunction
		forwardHidden2 C.CUfunction  // Added forwardHidden2 kernel
		forwardOutput  C.CUfunction
		forwardValue   C.CUfunction
		train          C.CUfunction
	}
	// Hyperparameters
	learningRate float32
	gamma        float32
	lambda       float32
	epsilon      float32
	// Optimizer state
	actorMomentum  []float32
	criticMomentum []float32
	actorRMS       []float32
	criticRMS      []float32
	scheduler      *LearningRateScheduler
	device         string // "cpu" or "cuda"
	// Training metadata
	totalSteps    int64
	metadataMutex sync.RWMutex
	// Context management
	contextLock sync.Mutex
	isContextCurrent bool
}

// NewModel creates a new model instance
func NewModel(gpuConfig *GPUConfig) *Model {
    m := &Model{
        device: "cuda",
        learningRate: 0.0001,  // Keep learning rate low for stability
        gamma: 0.99,
        lambda: 0.95,
        epsilon: 0.8,  // Increased from 0.4 to 0.8 for maximum exploration
    }

    // Initialize weights with Xavier/Glorot initialization
    // input->hidden1 (6->64)
    m.actorWeights = make([]float32, 6*64 + 64*64 + 64*3)  // input->hidden1 + hidden1->hidden2 + hidden2->output
    m.criticWeights = make([]float32, 64*1)  // hidden2->value (64->1)
    
    // Xavier/Glorot initialization with gain=1.0 for better stability
    for i := range m.actorWeights {
        if i < 6*64 {  // input->hidden1
            m.actorWeights[i] = float32(rand.NormFloat64()) * float32(math.Sqrt(2.0/float64(6)))  // gain=1.0
        } else if i < 6*64 + 64*64 {  // hidden1->hidden2
            m.actorWeights[i] = float32(rand.NormFloat64()) * float32(math.Sqrt(2.0/float64(64)))
        } else {  // hidden2->output
            m.actorWeights[i] = float32(rand.NormFloat64()) * float32(math.Sqrt(2.0/float64(64)))
        }
    }
    
    for i := range m.criticWeights {
        m.criticWeights[i] = float32(rand.NormFloat64()) * float32(math.Sqrt(2.0/float64(64)))
    }

    // Initialize optimizer state
    m.actorMomentum = make([]float32, len(m.actorWeights))
    m.criticMomentum = make([]float32, len(m.criticWeights))
    m.actorRMS = make([]float32, len(m.actorWeights))
    m.criticRMS = make([]float32, len(m.criticWeights))

    if gpuConfig != nil && gpuConfig.IsAvailable {
        // Initialize CUDA context and resources
        if err := m.initializeCUDA(gpuConfig); err != nil {
            log.Printf("[ERROR] Failed to initialize CUDA: %v", err)
            m.device = "cpu"
        }
    } else {
        m.device = "cpu"
    }

    return m
}

// initializeCUDA initializes CUDA resources for the model
func (m *Model) initializeCUDA(gpuConfig *GPUConfig) error {
    // Create CUDA context
    ctx, err := initializeCUDA()
    if err != nil {
        return fmt.Errorf("failed to initialize CUDA context: %v", err)
    }
    m.cudaContext = *ctx

    // Create CUDA streams
    if err := result(C.cuStreamCreate(&m.cudaStreams.compute, 0)); err != nil {
        return fmt.Errorf("failed to create compute stream: %v", err)
    }
    if err := result(C.cuStreamCreate(&m.cudaStreams.memory, 0)); err != nil {
        return fmt.Errorf("failed to create memory stream: %v", err)
    }

    // Load CUDA kernels
    if err := m.loadKernels(); err != nil {
        return fmt.Errorf("failed to load CUDA kernels: %v", err)
    }

    // Allocate GPU memory for weights
    if err := result(C.cuMemAlloc(&m.gpuBuffers.actorWeights, C.size_t(len(m.actorWeights)*4))); err != nil {
        return fmt.Errorf("failed to allocate GPU memory for actor weights: %v", err)
    }
    if err := result(C.cuMemAlloc(&m.gpuBuffers.criticWeights, C.size_t(len(m.criticWeights)*4))); err != nil {
        return fmt.Errorf("failed to allocate GPU memory for critic weights: %v", err)
    }

    // Copy weights to GPU
    if err := result(C.cuMemcpyHtoD(m.gpuBuffers.actorWeights, unsafe.Pointer(&m.actorWeights[0]), C.size_t(len(m.actorWeights)*4))); err != nil {
        return fmt.Errorf("failed to copy actor weights to GPU: %v", err)
    }
    if err := result(C.cuMemcpyHtoD(m.gpuBuffers.criticWeights, unsafe.Pointer(&m.criticWeights[0]), C.size_t(len(m.criticWeights)*4))); err != nil {
        return fmt.Errorf("failed to copy critic weights to GPU: %v", err)
    }

    // Allocate GPU memory for intermediate buffers
    if err := result(C.cuMemAlloc(&m.gpuBuffers.input, C.size_t(6*4))); err != nil { // 6 input features
        return fmt.Errorf("failed to allocate GPU memory for input buffer: %v", err)
    }
    if err := result(C.cuMemAlloc(&m.gpuBuffers.hidden, C.size_t(64*4))); err != nil { // 64 hidden units
        return fmt.Errorf("failed to allocate GPU memory for hidden buffer: %v", err)
    }
    if err := result(C.cuMemAlloc(&m.gpuBuffers.hidden2, C.size_t(64*4))); err != nil { // 64 hidden units
        return fmt.Errorf("failed to allocate GPU memory for hidden2 buffer: %v", err)
    }
    if err := result(C.cuMemAlloc(&m.gpuBuffers.output, C.size_t(3*4))); err != nil { // 3 output actions
        return fmt.Errorf("failed to allocate GPU memory for output buffer: %v", err)
    }
    if err := result(C.cuMemAlloc(&m.gpuBuffers.value, C.size_t(4))); err != nil { // 1 value output
        return fmt.Errorf("failed to allocate GPU memory for value buffer: %v", err)
    }

    return nil
}

// loadKernels loads the CUDA kernels
func (m *Model) loadKernels() error {
    // Create CUDA module from PTX file
    var module C.CUmodule
    ptxPath := "kernels.ptx"
    
    // Read PTX file
    ptxData, err := os.ReadFile(ptxPath)
    if err != nil {
        return fmt.Errorf("failed to read PTX file: %v", err)
    }
    
    // Allocate C memory for PTX data
    cPtxData := C.malloc(C.size_t(len(ptxData)))
    if cPtxData == nil {
        return fmt.Errorf("failed to allocate memory for PTX data")
    }
    defer C.free(cPtxData)
    
    // Copy PTX data to C memory
    C.memcpy(cPtxData, unsafe.Pointer(&ptxData[0]), C.size_t(len(ptxData)))
    
    // Load module
    if err := result(C.cuModuleLoadData(&module, cPtxData)); err != nil {
        return fmt.Errorf("failed to load CUDA module: %v", err)
    }

    // Get kernel functions
    kernelNames := []string{"forward_hidden1", "forward_hidden2", "forward_output", "forward_value", "train"}
    kernelFuncs := []*C.CUfunction{&m.kernels.forwardHidden, &m.kernels.forwardHidden2, &m.kernels.forwardOutput, &m.kernels.forwardValue, &m.kernels.train}
    
    for i, name := range kernelNames {
        // Allocate C memory for kernel name
        namePtr := C.CString(name)
        defer C.free(unsafe.Pointer(namePtr))
        
        var cKernelFunc C.CUfunction
        if err := result(C.cuModuleGetFunction(&cKernelFunc, module, namePtr)); err != nil {
            return fmt.Errorf("failed to get %s kernel: %v", name, err)
        }
        *kernelFuncs[i] = cKernelFunc
    }

    return nil
}

// Close releases all GPU resources
func (m *Model) Close() {
	if m.device == "cuda" {
		// Free pre-allocated buffers
		if m.gpuBuffers.input != 0 {
			C.cuMemFree(m.gpuBuffers.input)
		}
		if m.gpuBuffers.hidden != 0 {
			C.cuMemFree(m.gpuBuffers.hidden)
		}
		if m.gpuBuffers.output != 0 {
			C.cuMemFree(m.gpuBuffers.output)
		}
		if m.gpuBuffers.value != 0 {
			C.cuMemFree(m.gpuBuffers.value)
		}
		if m.gpuBuffers.actorWeights != 0 {
			C.cuMemFree(m.gpuBuffers.actorWeights)
		}
		if m.gpuBuffers.criticWeights != 0 {
			C.cuMemFree(m.gpuBuffers.criticWeights)
		}

		// Destroy streams
		if m.cudaStreams.compute != nil {
			C.cuStreamDestroy(m.cudaStreams.compute)
		}
		if m.cudaStreams.memory != nil {
			C.cuStreamDestroy(m.cudaStreams.memory)
		}

		// Destroy context
		if m.cudaContext != nil {
			cleanupCUDAContext(m.cudaContext)
		}
	}
}

// kernelArgs safely allocates C memory for kernel arguments
func kernelArgs(args []unsafe.Pointer) unsafe.Pointer {
    // Allocate C memory for the arguments
    size := uintptr(len(args)) * unsafe.Sizeof(args[0])
    cArgs := C.malloc(C.size_t(size))
    if cArgs == nil {
        return nil
    }
    
    // Copy arguments to C memory
    argArray := (*[1 << 30]unsafe.Pointer)(cArgs)[:len(args):len(args)]
    copy(argArray, args)
    
    return cArgs
}

// Forward performs a forward pass through the network
func (m *Model) Forward(obs []float32) ([]float32, float32) {
    if len(obs) != 6 {
        log.Printf("[ERROR] Invalid observation size: %d, expected 6", len(obs))
        return []float32{0, 0, 0}, 0
    }

    // Log input observations
    log.Printf("[DEBUG] Forward pass input - Observation: %v", obs)

    if m.device == "cuda" {
        // Copy input to GPU (observations are already normalized from Unity)
        if err := result(C.cuMemcpyHtoDAsync(m.gpuBuffers.input, unsafe.Pointer(&obs[0]), C.size_t(len(obs)*4), m.cudaStreams.memory)); err != nil {
            log.Printf("[ERROR] Failed to copy input to GPU: %v", err)
            return m.forwardCPU(obs)
        }

        // Launch kernels for each layer
        // Hidden layer 1
        args := make([]unsafe.Pointer, 5)
        args[0] = unsafe.Pointer(&m.gpuBuffers.input)
        args[1] = unsafe.Pointer(&m.gpuBuffers.actorWeights)
        args[2] = unsafe.Pointer(&m.gpuBuffers.hidden)
        inputSize := C.int(6)
        hiddenSize := C.int(64)
        args[3] = unsafe.Pointer(&inputSize)
        args[4] = unsafe.Pointer(&hiddenSize)
        cArgs := kernelArgs(args)
        if err := result(C.cuLaunchKernel(
            m.kernels.forwardHidden,
            C.uint((64+255)/256), 1, 1,
            256, 1, 1,
            0, m.cudaStreams.compute,
            (*unsafe.Pointer)(cArgs), nil,
        )); err != nil {
            C.free(cArgs)
            return m.forwardCPU(obs)
        }
        C.free(cArgs)

        // Hidden layer 2
        args[0] = unsafe.Pointer(&m.gpuBuffers.hidden)
        args[1] = unsafe.Pointer(&m.gpuBuffers.actorWeights)
        args[2] = unsafe.Pointer(&m.gpuBuffers.hidden2)
        args[3] = unsafe.Pointer(&hiddenSize)
        cArgs = kernelArgs(args)
        if err := result(C.cuLaunchKernel(
            m.kernels.forwardHidden2,
            C.uint((64+255)/256), 1, 1,
            256, 1, 1,
            0, m.cudaStreams.compute,
            (*unsafe.Pointer)(cArgs), nil,
        )); err != nil {
            C.free(cArgs)
            return m.forwardCPU(obs)
        }
        C.free(cArgs)

        // Output layer
        args[0] = unsafe.Pointer(&m.gpuBuffers.hidden2)
        args[1] = unsafe.Pointer(&m.gpuBuffers.actorWeights)
        args[2] = unsafe.Pointer(&m.gpuBuffers.output)
        outputSize := C.int(3)
        args[3] = unsafe.Pointer(&hiddenSize)
        args[4] = unsafe.Pointer(&outputSize)
        cArgs = kernelArgs(args)
        if err := result(C.cuLaunchKernel(
            m.kernels.forwardOutput,
            C.uint((3+255)/256), 1, 1,
            256, 1, 1,
            0, m.cudaStreams.compute,
            (*unsafe.Pointer)(cArgs), nil,
        )); err != nil {
            C.free(cArgs)
            return m.forwardCPU(obs)
        }
        C.free(cArgs)

        // Value prediction
        args[0] = unsafe.Pointer(&m.gpuBuffers.hidden2)
        args[1] = unsafe.Pointer(&m.gpuBuffers.criticWeights)
        args[2] = unsafe.Pointer(&m.gpuBuffers.value)
        args[3] = unsafe.Pointer(&hiddenSize)
        cArgs = kernelArgs(args)
        if err := result(C.cuLaunchKernel(
            m.kernels.forwardValue,
            1, 1, 1,
            1, 1, 1,
            0, m.cudaStreams.compute,
            (*unsafe.Pointer)(cArgs), nil,
        )); err != nil {
            C.free(cArgs)
            return m.forwardCPU(obs)
        }
        C.free(cArgs)

        // Synchronize and get results
        if err := result(C.cuStreamSynchronize(m.cudaStreams.compute)); err != nil {
            return m.forwardCPU(obs)
        }

        actions := make([]float32, 3)
        var valueResult float32
        if err := result(C.cuMemcpyDtoH(unsafe.Pointer(&actions[0]), m.gpuBuffers.output, C.size_t(3*4))); err != nil {
            return m.forwardCPU(obs)
        }
        if err := result(C.cuMemcpyDtoH(unsafe.Pointer(&valueResult), m.gpuBuffers.value, C.size_t(4))); err != nil {
            return m.forwardCPU(obs)
        }

        // Log raw actions before clamping
        log.Printf("[DEBUG] Raw actions before clamping: %v", actions)
        
        // Clamp actions to valid ranges
        for i := range actions {
            if i < 2 {  // Movement actions
                actions[i] = float32(math.Max(-1, math.Min(1, float64(actions[i]))))
            } else {  // Shooting action
                actions[i] = float32(math.Max(0, math.Min(1, float64(actions[i]))))
            }
        }

        return actions, valueResult
    }

    return m.forwardCPU(obs)
}

// forwardCPU performs a forward pass on CPU
func (m *Model) forwardCPU(obs []float32) ([]float32, float32) {
    if len(obs) != 6 {
        log.Printf("[ERROR] Invalid observation size: %d, expected 6", len(obs))
        return []float32{0, 0, 0}, 0
    }

    if len(m.actorWeights) == 0 || len(m.criticWeights) == 0 {
        log.Printf("[ERROR] Model weights not initialized")
        return []float32{0, 0, 0}, 0
    }

    // Log input observations for debugging
    log.Printf("[DEBUG] Forward pass input - Observation: %v", obs)

    // Hidden layer 1 (6->64)
    hidden1 := make([]float32, 64)
    for i := 0; i < 64; i++ {
        var sum float32
        // Access weights in column-major order to match CUDA kernel
        for j := 0; j < 6; j++ {
            sum += obs[j] * m.actorWeights[j*64+i]  // j*64+i for column-major
        }
        // ReLU activation
        hidden1[i] = float32(math.Max(0, float64(sum)))
    }

    // Log hidden1 activations
    log.Printf("[DEBUG] Hidden1 activations - Mean: %f, Max: %f", 
        mean(hidden1), max(hidden1))

    // Hidden layer 2 (64->64)
    hidden2 := make([]float32, 64)
    for i := 0; i < 64; i++ {
        var sum float32
        // Access weights in column-major order
        for j := 0; j < 64; j++ {
            sum += hidden1[j] * m.actorWeights[6*64+j*64+i]  // 6*64 offset for second layer
        }
        // ReLU activation
        hidden2[i] = float32(math.Max(0, float64(sum)))
    }

    // Log hidden2 activations
    log.Printf("[DEBUG] Hidden2 activations - Mean: %f, Max: %f", 
        mean(hidden2), max(hidden2))
    
    // Output layer (64->3)
    actions := make([]float32, 3)
    for i := 0; i < 3; i++ {
        var sum float32
        // Access weights in column-major order
        for j := 0; j < 64; j++ {
            sum += hidden2[j] * m.actorWeights[6*64+64*64+j*3+i]  // 6*64+64*64 offset for output layer
        }
        
        if i < 2 {  // Movement actions (moveX, moveZ)
            // Tanh activation for bounded movement [-1, 1]
            actions[i] = float32(math.Tanh(float64(sum)))
        } else {  // Shooting action
            // Sigmoid activation for shooting [0, 1]
            actions[i] = float32(1.0 / (1.0 + math.Exp(-float64(sum))))
        }
    }
    
    // Log raw actions before clamping
    log.Printf("[DEBUG] Raw actions before clamping: %v", actions)
    
    // Clamp actions to valid ranges
    for i := range actions {
        if i < 2 {  // Movement actions
            actions[i] = float32(math.Max(-1, math.Min(1, float64(actions[i]))))
        } else {  // Shooting action
            actions[i] = float32(math.Max(0, math.Min(1, float64(actions[i]))))
        }
    }
    
    // Log final actions
    log.Printf("[DEBUG] Final actions after clamping: %v", actions)
    
    // Value prediction (64->1)
    var value float32
    for i := 0; i < 64; i++ {
        value += hidden2[i] * m.criticWeights[i]
    }
    
    // Log value prediction
    log.Printf("[DEBUG] Value prediction: %f", value)
    
    return actions, value
}

// Helper functions for logging
func mean(values []float32) float32 {
    var sum float32
    for _, v := range values {
        sum += v
    }
    return sum / float32(len(values))
}

func max(values []float32) float32 {
    maxVal := values[0]
    for _, v := range values {
        if v > maxVal {
            maxVal = v
        }
    }
    return maxVal
}

// computeGAE computes Generalized Advantage Estimation
func computeGAE(rewards []float32, values []float32, dones []bool, nextValue float32, gamma float32, lambda float32) []float32 {
	advantages := make([]float32, len(rewards))
	gae := float32(0)
	
	for t := len(rewards) - 1; t >= 0; t-- {
		if t == len(rewards)-1 {
			nextValue = nextValue
		} else {
			nextValue = values[t+1]
		}
		
		delta := rewards[t] + gamma*nextValue*(1-float32(btoi(dones[t]))) - values[t]
		gae = delta + gamma*lambda*(1-float32(btoi(dones[t])))*gae
		advantages[t] = gae
	}
	
	// Normalize advantages
	mean := float32(0)
	for _, adv := range advantages {
		mean += adv
	}
	mean /= float32(len(advantages))
	
	std := float32(0)
	for _, adv := range advantages {
		std += (adv - mean) * (adv - mean)
	}
	std = float32(math.Sqrt(float64(std/float32(len(advantages)) + 1e-8)))
	
	for i := range advantages {
		advantages[i] = (advantages[i] - mean) / std
	}
	
	return advantages
}

// btoi converts bool to int
func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

// clipGradients clips gradients to prevent exploding gradients
func clipGradients(gradients []float32, maxNorm float32) {
    // Calculate L2 norm
    var norm float32
    for _, g := range gradients {
        norm += g * g
    }
    norm = float32(math.Sqrt(float64(norm)))

    // Clip if norm exceeds maxNorm
    if norm > maxNorm {
        scale := maxNorm / norm
        for i := range gradients {
            gradients[i] *= scale
        }
    }
}

// Train performs a training step on a batch of experiences
func (m *Model) Train(batch []Experience) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    if m.device == "cuda" {
        contextMutex.Lock()
        if err := makeContextCurrent(m.cudaContext); err != nil {
            log.Printf("[ERROR] Failed to set CUDA context in Train: %v", err)
            contextMutex.Unlock()
            return
        }
        contextMutex.Unlock()
        
        log.Printf("[DEBUG] Performing GPU training step with batch size: %d", len(batch))
        
        // Allocate GPU memory for gradients
        var actorGradients, criticGradients C.CUdeviceptr
        if err := result(C.cuMemAlloc(&actorGradients, C.size_t(len(m.actorWeights)*4))); err != nil {
            log.Printf("[ERROR] Failed to allocate GPU memory for actor gradients: %v", err)
            return
        }
        defer C.cuMemFree(actorGradients)

        if err := result(C.cuMemAlloc(&criticGradients, C.size_t(len(m.criticWeights)*4))); err != nil {
            log.Printf("[ERROR] Failed to allocate GPU memory for critic gradients: %v", err)
            return
        }
        defer C.cuMemFree(criticGradients)

        // Initialize gradients to zero
        zeroActorGradients := make([]float32, len(m.actorWeights))
        zeroCriticGradients := make([]float32, len(m.criticWeights))
        if err := result(C.cuMemcpyHtoD(actorGradients, unsafe.Pointer(&zeroActorGradients[0]), C.size_t(len(m.actorWeights)*4))); err != nil {
            log.Printf("[ERROR] Failed to initialize actor gradients: %v", err)
            return
        }
        if err := result(C.cuMemcpyHtoD(criticGradients, unsafe.Pointer(&zeroCriticGradients[0]), C.size_t(len(m.criticWeights)*4))); err != nil {
            log.Printf("[ERROR] Failed to initialize critic gradients: %v", err)
            return
        }

        // Process each experience in the batch
        for _, exp := range batch {
            // Forward pass
            action, value := m.Forward(exp.Observation)
            
            // Log forward pass results
            log.Printf("[DEBUG] Forward pass - Action: %v, Value: %f", action, value)

            // Calculate advantages and returns
            advantage := exp.Reward + m.gamma*exp.NextValue - value
            returns := exp.Reward + m.gamma*exp.NextValue
            
            // Log advantage calculation
            log.Printf("[DEBUG] Advantage calculation - Reward: %f, NextValue: %f, Value: %f, Advantage: %f",
                exp.Reward, exp.NextValue, value, advantage)

            // Calculate policy loss with PPO clipping
            policyLoss := float32(0)
            entropyBonus := float32(0)
            
            // Calculate action probabilities and log probabilities
            actionProbs := make([]float32, 3)
            logProbs := make([]float32, 3)
            
            for i := range action {
                if i < 2 { // Movement actions (tanh)
                    actionProbs[i] = (action[i] + 1) / 2 // Convert [-1,1] to [0,1]
                    logProbs[i] = float32(math.Log(float64(actionProbs[i])))
                } else { // Shooting action (sigmoid)
                    actionProbs[i] = action[i]
                    logProbs[i] = float32(math.Log(float64(actionProbs[i])))
                }
                
                // Add entropy bonus for exploration
                entropyBonus -= actionProbs[i] * logProbs[i]
            }
            
            // Log action probabilities
            log.Printf("[DEBUG] Action probabilities: %v, Log probabilities: %v", actionProbs, logProbs)
            
            // Calculate policy gradient with PPO clipping
            ratio := float32(math.Exp(float64(logProbs[0] - exp.LogProb)))
            clippedRatio := float32(math.Max(float64(1-m.epsilon*2.0), math.Min(float64(1+m.epsilon*2.0), float64(ratio))))  // Doubled clipping range
            
            policyLoss = -float32(math.Min(
                float64(ratio*advantage),
                float64(clippedRatio*advantage),
            ))
            
            // Add entropy bonus to encourage exploration
            policyLoss -= 0.2 * entropyBonus  // Increased from 0.05 to 0.2 for maximum entropy
            
            // Log policy loss components
            log.Printf("[DEBUG] Policy loss - Ratio: %f, ClippedRatio: %f, EntropyBonus: %f, TotalLoss: %f",
                ratio, clippedRatio, entropyBonus, policyLoss)

            // Calculate actor gradients with proper backpropagation
            actorGrads := make([]float32, len(m.actorWeights))
            
            // Backpropagate through the network
            // Output layer gradients
            outputGrads := make([]float32, 3)
            for i := range action {
                if i < 2 { // Movement actions
                    outputGrads[i] = -advantage * (1 - action[i]*action[i]) // tanh derivative
                } else { // Shooting action
                    outputGrads[i] = -advantage * action[i] * (1 - action[i]) // sigmoid derivative
                }
            }
            
            // Log output gradients
            log.Printf("[DEBUG] Output gradients: %v", outputGrads)
            
            // Hidden layer 2 gradients
            hidden2Grads := make([]float32, 64)
            for i := 0; i < 64; i++ {
                var grad float32
                for j := 0; j < 3; j++ {
                    grad += outputGrads[j] * m.actorWeights[6*64+64*64+i*3+j]
                }
                hidden2Grads[i] = grad * float32(math.Max(0, 1)) // ReLU derivative
            }
            
            // Hidden layer 1 gradients
            hidden1Grads := make([]float32, 64)
            for i := 0; i < 64; i++ {
                var grad float32
                for j := 0; j < 64; j++ {
                    grad += hidden2Grads[j] * m.actorWeights[6*64+i*64+j]
                }
                hidden1Grads[i] = grad * float32(math.Max(0, 1)) // ReLU derivative
            }
            
            // Update actor weights with column-major layout
            for i := 0; i < 64; i++ {
                for j := 0; j < 6; j++ {
                    actorGrads[j*64+i] = hidden1Grads[i] * exp.Observation[j]
                }
            }
            
            for i := 0; i < 64; i++ {
                for j := 0; j < 64; j++ {
                    actorGrads[6*64+i*64+j] = hidden2Grads[j] * float32(math.Max(0, float64(hidden1Grads[i])))
                }
            }
            
            for i := 0; i < 3; i++ {
                for j := 0; j < 64; j++ {
                    actorGrads[6*64+64*64+j*3+i] = outputGrads[i] * float32(math.Max(0, float64(hidden2Grads[j])))
                }
            }

            // Calculate critic gradients with proper value function loss
            valueLoss := float32(0.5) * float32(math.Pow(float64(value-returns), 2))
            criticGrad := value - returns
            
            // Log value loss
            log.Printf("[DEBUG] Value loss: %f", valueLoss)

            // Clip gradients
            clipGradients(actorGrads, 1.0)
            if criticGrad > 1.0 {
                criticGrad = 1.0
            } else if criticGrad < -1.0 {
                criticGrad = -1.0
            }

            // Copy gradients to GPU
            if err := result(C.cuMemcpyHtoD(actorGradients, unsafe.Pointer(&actorGrads[0]), C.size_t(len(actorGrads)*4))); err != nil {
                log.Printf("[ERROR] Failed to copy actor gradients to GPU: %v", err)
                return
            }
            if err := result(C.cuMemcpyHtoD(criticGradients, unsafe.Pointer(&criticGrad), C.size_t(4))); err != nil {
                log.Printf("[ERROR] Failed to copy critic gradient to GPU: %v", err)
                return
            }

            // Create variables for kernel arguments
            learningRate := C.float(m.learningRate)
            actorSize := C.int(len(m.actorWeights))
            criticSize := C.int(len(m.criticWeights))

            // Update actor weights using CUDA kernels
            args := []unsafe.Pointer{
                unsafe.Pointer(&m.gpuBuffers.actorWeights),
                unsafe.Pointer(&actorGradients),
                unsafe.Pointer(&learningRate),
                unsafe.Pointer(&actorSize),
            }
            if err := result(C.cuLaunchKernel(
                m.kernels.train,
                C.uint((len(m.actorWeights)+255)/256), 1, 1,
                256, 1, 1,
                0, m.cudaStreams.compute,
                &args[0], nil,
            )); err != nil {
                log.Printf("[ERROR] Failed to launch actor training kernel: %v", err)
                return
            }

            // Update critic weights using CUDA kernels
            args = []unsafe.Pointer{
                unsafe.Pointer(&m.gpuBuffers.criticWeights),
                unsafe.Pointer(&criticGradients),
                unsafe.Pointer(&learningRate),
                unsafe.Pointer(&criticSize),
            }
            if err := result(C.cuLaunchKernel(
                m.kernels.train,
                C.uint((len(m.criticWeights)+255)/256), 1, 1,
                256, 1, 1,
                0, m.cudaStreams.compute,
                &args[0], nil,
            )); err != nil {
                log.Printf("[ERROR] Failed to launch critic training kernel: %v", err)
                return
            }
        }

        // Copy updated weights back to CPU
        if err := result(C.cuMemcpyDtoH(unsafe.Pointer(&m.actorWeights[0]), m.gpuBuffers.actorWeights, C.size_t(len(m.actorWeights)*4))); err != nil {
            log.Printf("[ERROR] Failed to copy actor weights from GPU: %v", err)
            return
        }
        if err := result(C.cuMemcpyDtoH(unsafe.Pointer(&m.criticWeights[0]), m.gpuBuffers.criticWeights, C.size_t(len(m.criticWeights)*4))); err != nil {
            log.Printf("[ERROR] Failed to copy critic weights from GPU: %v", err)
            return
        }
    } else {
        // CPU implementation with the same logic
        log.Printf("[DEBUG] Performing CPU training step with batch size: %d", len(batch))
        for _, exp := range batch {
            // Forward pass
            action, value := m.Forward(exp.Observation)
            
            // Log forward pass results
            log.Printf("[DEBUG] Forward pass - Action: %v, Value: %f", action, value)

            // Calculate advantages and returns
            advantage := exp.Reward + m.gamma*exp.NextValue - value
            returns := exp.Reward + m.gamma*exp.NextValue
            
            // Log advantage calculation
            log.Printf("[DEBUG] Advantage calculation - Reward: %f, NextValue: %f, Value: %f, Advantage: %f",
                exp.Reward, exp.NextValue, value, advantage)

            // Calculate policy loss with PPO clipping
            policyLoss := float32(0)
            entropyBonus := float32(0)
            
            // Calculate action probabilities and log probabilities
            actionProbs := make([]float32, 3)
            logProbs := make([]float32, 3)
            
            for i := range action {
                if i < 2 { // Movement actions (tanh)
                    actionProbs[i] = (action[i] + 1) / 2 // Convert [-1,1] to [0,1]
                    logProbs[i] = float32(math.Log(float64(actionProbs[i])))
                } else { // Shooting action (sigmoid)
                    actionProbs[i] = action[i]
                    logProbs[i] = float32(math.Log(float64(actionProbs[i])))
                }
                
                // Add entropy bonus for exploration
                entropyBonus -= actionProbs[i] * logProbs[i]
            }
            
            // Log action probabilities
            log.Printf("[DEBUG] Action probabilities: %v, Log probabilities: %v", actionProbs, logProbs)
            
            // Calculate policy gradient with PPO clipping
            ratio := float32(math.Exp(float64(logProbs[0] - exp.LogProb)))
            clippedRatio := float32(math.Max(float64(1-m.epsilon*2.0), math.Min(float64(1+m.epsilon*2.0), float64(ratio))))  // Doubled clipping range
            
            policyLoss = -float32(math.Min(
                float64(ratio*advantage),
                float64(clippedRatio*advantage),
            ))
            
            // Add entropy bonus to encourage exploration
            policyLoss -= 0.2 * entropyBonus  // Increased from 0.05 to 0.2 for maximum entropy
            
            // Log policy loss components
            log.Printf("[DEBUG] Policy loss - Ratio: %f, ClippedRatio: %f, EntropyBonus: %f, TotalLoss: %f",
                ratio, clippedRatio, entropyBonus, policyLoss)

            // Calculate gradients
            actorGrads := make([]float32, len(m.actorWeights))
            criticGrads := make([]float32, len(m.criticWeights))

            // Backpropagate through the network
            // Output layer gradients
            outputGrads := make([]float32, 3)
            for i := range action {
                if i < 2 { // Movement actions
                    outputGrads[i] = -advantage * (1 - action[i]*action[i]) // tanh derivative
                } else { // Shooting action
                    outputGrads[i] = -advantage * action[i] * (1 - action[i]) // sigmoid derivative
                }
            }
            
            // Hidden layer 2 gradients
            hidden2Grads := make([]float32, 64)
            for i := 0; i < 64; i++ {
                var grad float32
                for j := 0; j < 3; j++ {
                    grad += outputGrads[j] * m.actorWeights[6*64+64*64+i*3+j]
                }
                hidden2Grads[i] = grad * float32(math.Max(0, 1)) // ReLU derivative
            }
            
            // Hidden layer 1 gradients
            hidden1Grads := make([]float32, 64)
            for i := 0; i < 64; i++ {
                var grad float32
                for j := 0; j < 64; j++ {
                    grad += hidden2Grads[j] * m.actorWeights[6*64+i*64+j]
                }
                hidden1Grads[i] = grad * float32(math.Max(0, 1)) // ReLU derivative
            }
            
            // Update actor weights with column-major layout
            for i := 0; i < 64; i++ {
                for j := 0; j < 6; j++ {
                    actorGrads[j*64+i] = hidden1Grads[i] * exp.Observation[j]
                }
            }
            
            for i := 0; i < 64; i++ {
                for j := 0; j < 64; j++ {
                    actorGrads[6*64+i*64+j] = hidden2Grads[j] * float32(math.Max(0, float64(hidden1Grads[i])))
                }
            }
            
            for i := 0; i < 3; i++ {
                for j := 0; j < 64; j++ {
                    actorGrads[6*64+64*64+j*3+i] = outputGrads[i] * float32(math.Max(0, float64(hidden2Grads[j])))
                }
            }

            // Calculate critic gradients
            for i := range m.criticWeights {
                criticGrads[i] = value - returns
            }

            // Clip gradients
            clipGradients(actorGrads, 1.0)
            clipGradients(criticGrads, 1.0)

            // Update weights with gradient descent
            for i := range m.actorWeights {
                m.actorWeights[i] -= m.learningRate * actorGrads[i]
            }
            for i := range m.criticWeights {
                m.criticWeights[i] -= m.learningRate * criticGrads[i]
            }
        }
    }

    // Update metadata
    m.metadataMutex.Lock()
    m.totalSteps += int64(len(batch))
    m.metadataMutex.Unlock()
}

// GetTotalSteps returns the total number of training steps
func (m *Model) GetTotalSteps() int64 {
	m.metadataMutex.RLock()
	defer m.metadataMutex.RUnlock()
	return m.totalSteps
}

// Save saves the model to a file
func (m *Model) Save(path string) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	data := struct {
		ActorWeights  []float32 `json:"actor_weights"`
		CriticWeights []float32 `json:"critic_weights"`
	}{
		ActorWeights:  m.actorWeights,
		CriticWeights: m.criticWeights,
	}
	
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	return json.NewEncoder(file).Encode(data)
}

// Load loads the model from a file
func (m *Model) Load(path string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	var data struct {
		ActorWeights  []float32 `json:"actor_weights"`
		CriticWeights []float32 `json:"critic_weights"`
	}
	
	if err := json.NewDecoder(file).Decode(&data); err != nil {
		return err
	}
	
	m.actorWeights = data.ActorWeights
	m.criticWeights = data.CriticWeights
	
	return nil
}

// Helper functions
func relu(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

// Helper functions for gradients
func reluGrad(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}

func tanhGrad(x float32) float32 {
	t := float32(math.Tanh(float64(x)))
	return 1 - t*t
}

// TrainingServer represents the main server structure
type TrainingServer struct {
	playerID        int
	port            int
	model           *Model
	buffer          *ExperienceBuffer
	metadata        *TrainingMetadata
	activeConns     int
	connMutex       sync.RWMutex
	bufferMutex     sync.RWMutex
	metadataMutex   sync.RWMutex
	shutdownChan    chan struct{}
	instanceData    map[string]map[string]interface{}
	instanceMutex   sync.RWMutex
	checkpointPath  string
	logPath         string
	lastCheckpoint  time.Time
	checkpointMutex sync.RWMutex
	logger          *Logger
}

// NewTrainingServer creates a new training server
func NewTrainingServer(playerID, port int) *TrainingServer {
	// Create directories
	os.MkdirAll(CheckpointDir, 0755)
	os.MkdirAll(LogDir, 0755)
	
	checkpointPath := filepath.Join(CheckpointDir, fmt.Sprintf("player_%d", playerID))
	logPath := filepath.Join(LogDir, fmt.Sprintf("player_%d.log", playerID))
	
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())
	
	logger, err := NewLogger(playerID, logPath)
	if err != nil {
		log.Printf("Error creating logger: %v", err)
	}

	// Initialize GPU first
	gpuConfig, err := initGPU()
	if err != nil {
		log.Printf("Warning: GPU initialization failed: %v. Falling back to CPU mode.", err)
		gpuConfig = &GPUConfig{IsAvailable: false}
	} else {
		log.Printf("GPU initialization successful for player %d", playerID)
	}
	
	// Create model with GPU config
	model := NewModel(gpuConfig)
	if model.device == "cuda" {
		log.Printf("Model initialized with GPU support for player %d", playerID)
	} else {
		log.Printf("Model initialized in CPU mode for player %d", playerID)
	}
	
	server := &TrainingServer{
		playerID:       playerID,
		port:           port,
		model:          model,
		buffer:         NewExperienceBuffer(BufferSize),
		metadata:       &TrainingMetadata{},
		shutdownChan:   make(chan struct{}),
		instanceData:   make(map[string]map[string]interface{}),
		checkpointPath: checkpointPath,
		logPath:        logPath,
		lastCheckpoint: time.Now(),
		logger:         logger,
	}
	
	// Load latest checkpoint if exists
	if err := server.loadLatestCheckpoint(); err != nil {
		log.Printf("No checkpoint found or error loading: %v", err)
	}
	
	return server
}

// loadLatestCheckpoint loads the most recent checkpoint
func (s *TrainingServer) loadLatestCheckpoint() error {
	files, err := filepath.Glob(s.checkpointPath + "_*.json")
	if err != nil {
		return err
	}
	
	if len(files) == 0 {
		return fmt.Errorf("no checkpoints found")
	}
	
	// Find latest checkpoint
	var latestFile string
	var latestTime time.Time
	for _, file := range files {
		info, err := os.Stat(file)
		if err != nil {
			continue
		}
		if info.ModTime().After(latestTime) {
			latestTime = info.ModTime()
			latestFile = file
		}
	}
	
	if latestFile == "" {
		return fmt.Errorf("no valid checkpoints found")
	}
	
	// Load checkpoint
	data, err := os.ReadFile(latestFile)
	if err != nil {
		return err
	}
	
	var checkpoint ModelCheckpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return err
	}
	
	// Restore model state
	s.metadataMutex.Lock()
	s.metadata.TotalSteps = checkpoint.TotalSteps
	s.metadata.TotalEpisodes = checkpoint.TotalEpisodes
	s.metadata.TotalReward = checkpoint.TotalReward
	s.metadata.LossHistory = checkpoint.LossHistory
	s.metadataMutex.Unlock()
	
	s.model.LoadWeights(checkpoint.Weights.Actor, checkpoint.Weights.Critic)
	
	log.Printf("Loaded checkpoint from %s", latestFile)
	return nil
}

// saveCheckpoint saves the current model state
func (s *TrainingServer) saveCheckpoint() error {
    s.checkpointMutex.Lock()
    defer s.checkpointMutex.Unlock()
    
    // Check if enough time has passed since last checkpoint
    if time.Since(s.lastCheckpoint) < 5*time.Minute {
        return nil
    }
    
    timestamp := time.Now().Format("20060102_150405")
    
    // Save model weights
    modelPath := filepath.Join(CheckpointDir, fmt.Sprintf("model_%s.pt", timestamp))
    if err := s.model.Save(modelPath); err != nil {
        return fmt.Errorf("failed to save model: %v", err)
    }
    
    // Save metadata separately
    metadataPath := filepath.Join(CheckpointDir, fmt.Sprintf("metadata_%s.json", timestamp))
    metadata := struct {
        Timestamp     string    `json:"timestamp"`
        TotalSteps    int64     `json:"total_steps"`
        TotalEpisodes int64     `json:"total_episodes"`
        TotalReward   float32   `json:"total_reward"`
        LossHistory   []float32 `json:"loss_history"`
    }{
        Timestamp:     timestamp,
        TotalSteps:    s.metadata.TotalSteps,
        TotalEpisodes: s.metadata.TotalEpisodes,
        TotalReward:   s.metadata.TotalReward,
        LossHistory:   s.metadata.LossHistory,
    }
    
    data, err := json.MarshalIndent(metadata, "", "  ")
    if err != nil {
        return fmt.Errorf("failed to marshal metadata: %v", err)
    }
    
    if err := os.WriteFile(metadataPath, data, 0644); err != nil {
        return fmt.Errorf("failed to write metadata: %v", err)
    }
    
    s.lastCheckpoint = time.Now()
    log.Printf("Saved model to %s and metadata to %s", modelPath, metadataPath)
    return nil
}

// monitorResources periodically checks system resources
func (s *TrainingServer) monitorResources() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-s.shutdownChan:
			return
		case <-ticker.C:
			if err := s.logResourceUsage(); err != nil {
				log.Printf("Error monitoring resources: %v", err)
			}
			
			// Save checkpoint periodically
			if err := s.saveCheckpoint(); err != nil {
				log.Printf("Error saving checkpoint: %v", err)
			}
		}
	}
}

// logResourceUsage logs current resource usage
func (s *TrainingServer) logResourceUsage() error {
	proc, err := process.NewProcess(int32(os.Getpid()))
	if err != nil {
		return fmt.Errorf("failed to get process: %v", err)
	}
	
	memInfo, err := proc.MemoryInfo()
	if err != nil {
		return fmt.Errorf("failed to get memory info: %v", err)
	}
	
	cpuPercent, err := proc.CPUPercent()
	if err != nil {
		return fmt.Errorf("failed to get CPU percent: %v", err)
	}
	
	openFiles, err := proc.OpenFiles()
	if err != nil {
		return fmt.Errorf("failed to get open files: %v", err)
	}
	
	s.connMutex.RLock()
	activeConns := s.activeConns
	s.connMutex.RUnlock()
	
	s.bufferMutex.RLock()
	bufferSize := s.buffer.Size()
	s.bufferMutex.RUnlock()
	
	stats := MemoryStats{
		Timestamp:        time.Now().Format(time.RFC3339),
		RSS:             memInfo.RSS,
		VMS:             memInfo.VMS,
		CPUPercent:      cpuPercent,
		ActiveConnections: activeConns,
		BufferSize:      bufferSize,
		OpenFiles:       len(openFiles),
	}
	
	s.metadataMutex.Lock()
	s.metadata.MemoryUsage = append(s.metadata.MemoryUsage, stats)
	if len(s.metadata.MemoryUsage) > 1000 {
		s.metadata.MemoryUsage = s.metadata.MemoryUsage[len(s.metadata.MemoryUsage)-1000:]
	}
	s.metadataMutex.Unlock()
	
	// Log to file
	logEntry := fmt.Sprintf("[%s] Active connections: %d/%d, Buffer: %d/%d, Memory: %.1fGB, CPU: %.1f%%, Files: %d\n",
		stats.Timestamp, activeConns, MaxConnections, bufferSize, BufferSize,
		float64(memInfo.RSS)/1024/1024/1024, cpuPercent, len(openFiles))
	
	if err := os.WriteFile(s.logPath, []byte(logEntry), 0644); err != nil {
		return fmt.Errorf("failed to write log file: %v", err)
	}
	
	return nil
}

// setupServerEnvironment configures system limits
func (s *TrainingServer) setupServerEnvironment() error {
	// Set file descriptor limits
	var rLimit syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &rLimit); err != nil {
		return fmt.Errorf("failed to get rlimit: %v", err)
	}
	
	rLimit.Cur = rLimit.Max
	if err := syscall.Setrlimit(syscall.RLIMIT_NOFILE, &rLimit); err != nil {
		return fmt.Errorf("failed to set rlimit: %v", err)
	}
	
	return nil
}

// setupRoutes configures the HTTP routes
func (s *TrainingServer) setupRoutes(r *gin.Engine) {
	// Health check
	r.GET("/health", s.handleHealth)
	
	// Action endpoint
	r.POST("/get_action", s.handleGetAction)
	
	// Reward update endpoint
	r.POST("/update_reward", s.handleUpdateReward)
	
	// Training endpoint
	r.POST("/train", s.handleTrain)
	
	// Save model endpoint
	r.POST("/save_model", s.handleSaveModel)
	
	// Shutdown endpoint
	r.POST("/shutdown", s.handleShutdown)
}

// handleHealth handles health check requests
func (s *TrainingServer) handleHealth(c *gin.Context) {
	s.connMutex.RLock()
	activeConns := s.activeConns
	s.connMutex.RUnlock()
	
	s.bufferMutex.RLock()
	bufferSize := s.buffer.Size()
	s.bufferMutex.RUnlock()
	
	s.metadataMutex.RLock()
	totalSteps := s.metadata.TotalSteps
	s.metadataMutex.RUnlock()
	
	// Count active instances per player
	s.instanceMutex.RLock()
	playerCounts := make(map[int]int)
	for _, data := range s.instanceData {
		if pid, ok := data["player_id"].(int); ok {
			playerCounts[pid]++
		}
	}
	s.instanceMutex.RUnlock()
	
	c.JSON(http.StatusOK, gin.H{
		"status":            "healthy",
		"port":             s.port,
		"buffer_size":      bufferSize,
		"total_steps":      totalSteps,
		"active_connections": activeConns,
		"active_instances": playerCounts,
		"gpu_enabled":      s.model.device == "cuda",
	})
}

// handleGetAction handles action requests
func (s *TrainingServer) handleGetAction(c *gin.Context) {
    startTime := time.Now()
    requestID := fmt.Sprintf("%d", time.Now().UnixNano())
    
    s.logger.Log("info", fmt.Sprintf("[mario] [%s] Starting get_action request", requestID))
    
    var req struct {
        Observation []float32 `json:"observation"`
        InstanceID  string    `json:"instance_id"`
        PlayerID    int       `json:"player_id"`
    }
    
    if err := c.BindJSON(&req); err != nil {
        s.logger.Log("error", fmt.Sprintf("[mario] [%s] JSON parsing failed: %v", requestID, err))
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
        return
    }

    s.logger.Log("debug", fmt.Sprintf("[mario] [%s] Request details - Player: %d, Instance: %s, Observation: %v", 
        requestID, req.PlayerID, req.InstanceID, req.Observation))
    
    // Forward pass
    action, value := s.model.Forward(req.Observation)
    
    s.logger.Log("info", fmt.Sprintf("[mario] [%s] Action response - Player: %d, Action: %v, Value: %f, Time: %v", 
        requestID, req.PlayerID, action, value, time.Since(startTime)))
    
    response := gin.H{
        "action": action,
        "value":  value,
    }
    
    c.JSON(http.StatusOK, response)
}

// handleUpdateReward handles reward updates
func (s *TrainingServer) handleUpdateReward(c *gin.Context) {
    s.logger.Log("info", "Processing update_reward request")
    
    var req struct {
        InstanceID string  `json:"instance_id"`
        Reward     float32 `json:"reward"`
        Done       bool    `json:"done"`
        PlayerID   int     `json:"player_id"`
        NextState  []float32 `json:"next_state"`  // Add next_state to request
    }
    
    if err := c.BindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
        return
    }
    
    // Get player ID from instance data if not provided
    if req.PlayerID == 0 {
        s.instanceMutex.RLock()
        if data, ok := s.instanceData[req.InstanceID]; ok {
            if pid, ok := data["player_id"].(int); ok {
                req.PlayerID = pid
            }
        }
        s.instanceMutex.RUnlock()
    }
    
    // Add experience to buffer with next state
    exp := Experience{
        Reward: req.Reward,
        Done:   req.Done,
        NextValue: 0,  // Will be updated when next state is processed
    }
    
    // Store next state for value prediction
    s.instanceMutex.Lock()
    if s.instanceData[req.InstanceID] == nil {
        s.instanceData[req.InstanceID] = make(map[string]interface{})
    }
    s.instanceData[req.InstanceID]["next_state"] = req.NextState
    s.instanceMutex.Unlock()
    
    s.bufferMutex.Lock()
    s.buffer.Add(exp, 1.0) // Default priority of 1.0
    s.bufferMutex.Unlock()
    
    if req.Done {
        s.metadataMutex.Lock()
        s.metadata.TotalEpisodes++
        s.metadataMutex.Unlock()
        
        // Clean up instance data when episode is done
        s.instanceMutex.Lock()
        delete(s.instanceData, req.InstanceID)
        s.instanceMutex.Unlock()
    }
    
    s.logger.Log("info", fmt.Sprintf("Reward updated: instance=%s, player=%d, reward=%f, done=%v", 
        req.InstanceID, req.PlayerID, req.Reward, req.Done))
    c.JSON(http.StatusOK, gin.H{"status": "success"})
}

// handleTrain handles training requests
func (s *TrainingServer) handleTrain(c *gin.Context) {
    s.logger.Log("info", "Processing train request")
    
    s.bufferMutex.RLock()
    batch, _ := s.buffer.GetBatch(BatchSize)
    s.bufferMutex.RUnlock()
    
    if batch == nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Insufficient buffer size"})
        return
    }
    
    s.model.Train(batch)
    
    s.metadataMutex.Lock()
    s.metadata.TotalSteps += int64(len(batch))
    s.metadataMutex.Unlock()
    
    c.JSON(http.StatusOK, gin.H{
        "status":         "success",
        "buffer_size":    s.buffer.Size(),
        "total_steps":    s.metadata.TotalSteps,
        "total_episodes": s.metadata.TotalEpisodes,
    })
    
    s.logger.Log("info", fmt.Sprintf("Training completed: buffer_size=%d", s.buffer.Size()))
}

// handleShutdown handles shutdown requests
func (s *TrainingServer) handleShutdown(c *gin.Context) {
	close(s.shutdownChan)
	c.JSON(http.StatusOK, gin.H{"status": "shutting_down"})
}

// handleSaveModel handles model saving requests
func (s *TrainingServer) handleSaveModel(c *gin.Context) {
    var req struct {
        Path string `json:"path"`
    }
    
    if err := c.BindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
        return
    }
    
    // Create directory if it doesn't exist
    dir := filepath.Dir(req.Path)
    if err := os.MkdirAll(dir, 0755); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to create directory: %v", err)})
        return
    }
    
    // Save the model
    if err := s.model.Save(req.Path); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to save model: %v", err)})
        return
    }
    
    s.logger.Log("info", fmt.Sprintf("Model saved to %s", req.Path))
    c.JSON(http.StatusOK, gin.H{"status": "success", "path": req.Path})
}

// Run starts the server
func (s *TrainingServer) Run() error {
	s.logger.Log("info", fmt.Sprintf("Starting server on port %d", s.port))
	
	if err := s.setupServerEnvironment(); err != nil {
		return fmt.Errorf("failed to setup server environment: %v", err)
	}
	
	// Start resource monitoring
	go s.monitorResources()
	
	// Configure Gin
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())
	
	// Setup routes
	s.setupRoutes(r)
	
	// Start server
	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", s.port),
		Handler: r,
	}
	
	// Handle graceful shutdown
	go func() {
		<-s.shutdownChan
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			log.Printf("Error shutting down server: %v", err)
		}
	}()
	
	defer s.logger.Close()
	
	log.Printf("Player %d server starting on port %d", s.playerID, s.port)
	return server.ListenAndServe()
}

// Logger provides enhanced logging capabilities
type Logger struct {
	mu sync.RWMutex
	file *os.File
	level string
	playerID int
}

// NewLogger creates a new logger
func NewLogger(playerID int, logPath string) (*Logger, error) {
	file, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, err
	}
	
	return &Logger{
		file: file,
		level: "info",
		playerID: playerID,
	}, nil
}

// Log writes a log message
func (l *Logger) Log(level, message string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	logEntry := fmt.Sprintf("[%s] [%s] [Player %d] %s\n", timestamp, level, l.playerID, message)
	
	if l.file != nil {
		l.file.WriteString(logEntry)
	}
	
	// Also log to stdout
	log.Print(logEntry)
}

// Close closes the logger
func (l *Logger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	
	if l.file != nil {
		return l.file.Close()
	}
	return nil
}

// Implement Model.LoadWeights and GetWeights
func (m *Model) LoadWeights(actor, critic []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()
	copy(m.actorWeights, actor)
	copy(m.criticWeights, critic)
}

func (m *Model) GetWeights() ([]float32, []float32) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	actor := make([]float32, len(m.actorWeights))
	critic := make([]float32, len(m.criticWeights))
	copy(actor, m.actorWeights)
	copy(critic, m.criticWeights)
	return actor, critic
}

// Implement Size() for ExperienceBuffer
func (b *ExperienceBuffer) Size() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.data)
}

type AccuracyData struct {
    PlayerID int     `json:"player_id"`
    Hit      bool    `json:"hit"`
    Damage   float32 `json:"damage"`
    Position struct {
        X float32 `json:"x"`
        Y float32 `json:"y"`
        Z float32 `json:"z"`
    } `json:"position"`
}

func (s *TrainingServer) handleAccuracy(w http.ResponseWriter, r *http.Request) {
    var data AccuracyData
    if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    // Log accuracy data
    s.logger.Log("info", fmt.Sprintf("Accuracy data received - Player: %d, Hit: %v, Damage: %f", 
        data.PlayerID, data.Hit, data.Damage))

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{"status": "success"})
}

func (s *TrainingServer) Start() {
    // ... existing code ...
    
    // Add accuracy endpoint
    http.HandleFunc("/accuracy", s.handleAccuracy)
    
    // ... rest of existing code ...
}

func main() {
	// Set up logging to file
	logFile, err := os.OpenFile("server.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal("Failed to open log file:", err)
	}
	defer logFile.Close()
	log.SetOutput(logFile)
	
	// Set GOMAXPROCS
	runtime.GOMAXPROCS(NumWorkers)
	
	// Create required directories
	os.MkdirAll(CheckpointDir, 0755)
	os.MkdirAll(LogDir, 0755)
	
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())
	
	// Create single server instance for both players
	log.Printf("Creating shared server on port 5000")
	server := NewTrainingServer(0, 5000) // Use player_id 0 to indicate shared server
	
	// Handle signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	
	// Start server
	go func() {
		log.Printf("Starting shared server on port 5000")
		if err := server.Run(); err != nil {
			log.Printf("Server error: %v", err)
		}
	}()
	
	// Wait for signal
	<-sigChan
	log.Println("Received shutdown signal")
	
	// Trigger shutdown
	close(server.shutdownChan)
	
	// Explicitly cleanup CUDA context
	server.model.Close()

	log.Println("Server stopped and CUDA context cleaned up")
} 