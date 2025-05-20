package main

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
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
)

const (
	NumWorkers     = 4
	BatchSize      = 1024
	BufferSize     = 1000000
	MinBufferSize  = 10000
	MaxConnections = 1000
	Port           = 5001
	LogFile        = "parallel_games.log"
)

type Experience struct {
	Observation []float32 `json:"observation"`
	Action      struct {
		MoveX   float32 `json:"move_x"`
		MoveZ   float32 `json:"move_z"`
		Rotate  float32 `json:"rotate"`
		Shoot   bool    `json:"shoot"`
	} `json:"action"`
	Reward      float32   `json:"reward"`
	Value       float32   `json:"value"`
	NextValue   float32   `json:"next_value"`
	LogProb     float32   `json:"log_prob"`
	Done        bool      `json:"done"`
}

type ExperienceBuffer struct {
	data    []Experience
	maxSize int
	mu      sync.RWMutex
}

func NewExperienceBuffer(maxSize int) *ExperienceBuffer {
	return &ExperienceBuffer{
		data:    make([]Experience, 0, maxSize),
		maxSize: maxSize,
	}
}

// func (b *ExperienceBuffer) Add(exp Experience) {
// 	b.mu.Lock()
// 	defer b.mu.Unlock()
// 	if len(b.data) >= b.maxSize {
// 		b.data = b.data[1:]
// 	}
// 	b.data = append(b.data, exp)
// 	log.Printf("[CPU Server] Added experience to buffer - Current size: %d/%d", len(b.data), b.maxSize)
// }

// func (b *ExperienceBuffer) GetBatch(size int) []Experience {
// 	b.mu.RLock()
// 	if len(b.data) < size {
// 		b.mu.RUnlock()
// 		return nil
// 	}
// 	// Create a copy of the data we need while holding the lock
// 	indices := rand.Perm(len(b.data))[:size]
// 	dataCopy := make([]Experience, size)
// 	for i, idx := range indices {
// 		dataCopy[i] = b.data[idx]
// 	}
// 	b.mu.RUnlock()
// 	return dataCopy
// }

func (b *ExperienceBuffer) Size() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.data)
}

type Model struct {
	mu           sync.RWMutex
	actorWeights  []float32
	criticWeights []float32
	learningRate  float32
	gamma         float32
	lambda        float32
	totalSteps    int64
}

func NewModel() *Model {
	m := &Model{
		learningRate: 0.0001,
		gamma:       0.99,
		lambda:      0.95,
	}
	// Input size is now 308 (8 basic + 300 grid values)
	m.actorWeights = make([]float32, 308*64+64*64+64*4)
	m.criticWeights = make([]float32, 64*1)
	for i := range m.actorWeights {
		m.actorWeights[i] = float32(rand.NormFloat64()) * float32(math.Sqrt(2.0/308))
	}
	for i := range m.criticWeights {
		m.criticWeights[i] = float32(rand.NormFloat64()) * float32(math.Sqrt(2.0/64))
	}
	return m
}

func (m *Model) Forward(obs []float32) ([]float32, float32) {
	if len(obs) != 308 {
		return []float32{0, 0, 0, 0}, 0
	}
	// Hidden layer 1 (308->64)
	hidden1 := make([]float32, 64)
	for i := 0; i < 64; i++ {
		var sum float32
		for j := 0; j < 308; j++ {
			sum += obs[j] * m.actorWeights[j*64+i]
		}
		hidden1[i] = float32(math.Max(0, float64(sum)))
	}
	// Hidden layer 2 (64->64)
	hidden2 := make([]float32, 64)
	for i := 0; i < 64; i++ {
		var sum float32
		for j := 0; j < 64; j++ {
			sum += hidden1[j] * m.actorWeights[308*64+j*64+i]
		}
		hidden2[i] = float32(math.Max(0, float64(sum)))
	}
	// Output layer (64->4)
	actions := make([]float32, 4)
	for i := 0; i < 4; i++ {
		var sum float32
		for j := 0; j < 64; j++ {
			sum += hidden2[j] * m.actorWeights[308*64+64*64+j*4+i]
		}
		if i < 3 {  // First 3 actions (move_x, move_z, rotate) use tanh
			actions[i] = float32(math.Tanh(float64(sum)))
		} else {    // Last action (shoot) uses sigmoid
			actions[i] = float32(1.0 / (1.0 + math.Exp(-float64(sum))))
		}
	}
	// Value prediction (64->1)
	var value float32
	for i := 0; i < 64; i++ {
		value += hidden2[i] * m.criticWeights[i]
	}
	return actions, value
}

// Helper function to convert bool to int
func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

// computeGAE calculates Generalized Advantage Estimation
func computeGAE(rewards []float32, values []float32, dones []bool, nextValue float32, gamma float32, lambda float32) []float32 {
	advantages := make([]float32, len(rewards))
	
	for t := len(rewards) - 1; t >= 0; t-- {
		if t == len(rewards)-1 {
			nextValue = values[t]
		}
		delta := rewards[t] + gamma*nextValue*(1-float32(btoi(dones[t]))) - values[t]
		if t < len(rewards)-1 {
			advantages[t] = delta + gamma*lambda*advantages[t+1]
		} else {
			advantages[t] = delta
		}
	}
	return advantages
}

func (m *Model) Train(batch []Experience) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	log.Printf("[CPU Server] Starting training with batch size: %d", len(batch))
	
	// Extract sequences for GAE computation
	rewards := make([]float32, len(batch))
	values := make([]float32, len(batch))
	dones := make([]bool, len(batch))
	
	// First pass: compute values and collect data
	for i, exp := range batch {
		_, value := m.Forward(exp.Observation)
		values[i] = value
		
		// Log rewards for debugging
		// mario commenting reward logs
		// if i%100 == 0 {
		// 	log.Printf("[CPU Server] Sample %d - Base reward: %.4f", i+1, exp.Reward)
		// }
		
		rewards[i] = exp.Reward
		dones[i] = exp.Done
	}
	
	// Compute advantages using GAE
	advantages := computeGAE(rewards, values, dones, values[len(values)-1], m.gamma, m.lambda)
	
	var totalAdvantage float32
	var totalValueLoss float32
	var totalPolicyLoss float32
	
	// Second pass: update model using computed advantages
	for i, exp := range batch {
		action, value := m.Forward(exp.Observation)
		advantage := advantages[i]
		returns := rewards[i] + m.gamma*exp.NextValue
		
		// Log every 100th experience for detailed debugging
		// mario commenting reward
		// if i%100 == 0 {
		// 	log.Printf("[CPU Server] Training sample %d/%d - Reward: %.4f, Value: %.4f, Advantage: %.4f", 
		// 		i+1, len(batch), rewards[i], value, advantage)
		// }
		
		actorGrads := make([]float32, len(m.actorWeights))
		criticGrads := make([]float32, len(m.criticWeights))
		outputGrads := make([]float32, 4)
		
		// Calculate policy gradients
		for i := range action {
			if i < 3 {
				outputGrads[i] = -advantage * (1 - action[i]*action[i])
			} else {
				outputGrads[i] = -advantage * action[i] * (1 - action[i])
			}
		}
		
		hidden2Grads := make([]float32, 64)
		for i := 0; i < 64; i++ {
			var grad float32
			for j := 0; j < 4; j++ {
				grad += outputGrads[j] * m.actorWeights[308*64+64*64+i*4+j]
			}
			hidden2Grads[i] = grad
		}
		
		hidden1Grads := make([]float32, 64)
		for i := 0; i < 64; i++ {
			var grad float32
			for j := 0; j < 64; j++ {
				grad += hidden2Grads[j] * m.actorWeights[308*64+i*64+j]
			}
			hidden1Grads[i] = grad
		}
		
		// Calculate actor gradients
		for i := 0; i < 64; i++ {
			for j := 0; j < 308; j++ {
				actorGrads[j*64+i] = hidden1Grads[i] * exp.Observation[j]
			}
		}
		for i := 0; i < 64; i++ {
			for j := 0; j < 64; j++ {
				actorGrads[308*64+i*64+j] = hidden2Grads[j]
			}
		}
		for i := 0; i < 4; i++ {
			for j := 0; j < 64; j++ {
				actorGrads[308*64+64*64+j*4+i] = outputGrads[i]
			}
		}
		
		// Calculate critic gradients
		for i := range m.criticWeights {
			criticGrads[i] = value - returns
		}
		
		// Update weights
		for i := range m.actorWeights {
			m.actorWeights[i] -= m.learningRate * actorGrads[i]
		}
		for i := range m.criticWeights {
			m.criticWeights[i] -= m.learningRate * criticGrads[i]
		}
		
		// Accumulate metrics
		totalAdvantage += advantage
		totalValueLoss += float32(math.Abs(float64(value - returns)))
		totalPolicyLoss += float32(math.Abs(float64(advantage)))
	}
	
	// Calculate average metrics
	avgAdvantage := totalAdvantage / float32(len(batch))
	avgValueLoss := totalValueLoss / float32(len(batch))
	avgPolicyLoss := totalPolicyLoss / float32(len(batch))
	
	m.totalSteps += int64(len(batch))
	
	// Log training summary
	log.Printf("[CPU Server] Training completed - Total steps: %d, Avg Advantage: %.4f, Avg Value Loss: %.4f, Avg Policy Loss: %.4f",
		m.totalSteps, avgAdvantage, avgValueLoss, avgPolicyLoss)
}

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
	copy(m.actorWeights, data.ActorWeights)
	copy(m.criticWeights, data.CriticWeights)
	return nil
}

type TrainingServer struct {
	port        int
	model       *Model
	buffer      *ExperienceBuffer
	activeConns int
	connMutex   sync.RWMutex
	shutdownChan chan struct{}
	metadata    struct {
		TotalReward   float32
		RewardHistory []float32
	}
	metadataMutex sync.RWMutex
}

func NewTrainingServer(port int) *TrainingServer {
	model := NewModel()
	return &TrainingServer{
		port:        port,
		model:       model,
		buffer:      NewExperienceBuffer(BufferSize),
		shutdownChan: make(chan struct{}),
	}
}

func (s *TrainingServer) handleGetAction(c *gin.Context) {
	var req struct {
		Observation []float32 `json:"observation"`
		InstanceID  string    `json:"instance_id"`
		PlayerID    int       `json:"player_id"`
	}
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}
	action, value := s.model.Forward(req.Observation)
	c.JSON(http.StatusOK, gin.H{
		"action": action,
		"value":  value,
	})
}

func (s *TrainingServer) handleTrain(c *gin.Context) {
	var req struct {
		Experiences []Experience `json:"experiences"`
	}
	if err := c.BindJSON(&req); err != nil {
		log.Printf("[CPU Server] Error parsing training request: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}
	
	if len(req.Experiences) == 0 {
		log.Printf("[CPU Server] Error: No experiences provided for training")
		c.JSON(http.StatusBadRequest, gin.H{"error": "No experiences provided"})
		return
	}
	
	log.Printf("[CPU Server] Received %d experiences for training", len(req.Experiences))
	
	// Train directly on the experiences
	s.model.Train(req.Experiences)
	
	log.Printf("[CPU Server] Training completed on %d experiences", len(req.Experiences))
	
	c.JSON(http.StatusOK, gin.H{
		"status":            "success",
		"trained_experiences": len(req.Experiences),
	})
}

func (s *TrainingServer) handleSaveModel(c *gin.Context) {
	var req struct {
		Path string `json:"path"`
	}
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}
	dir := filepath.Dir(req.Path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to create directory: %v", err)})
		return
	}
	if err := s.model.Save(req.Path); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to save model: %v", err)})
		return
	}
	c.JSON(http.StatusOK, gin.H{"status": "success", "path": req.Path})
}

func (s *TrainingServer) handleUpdateReward(c *gin.Context) {
	var req struct {
		InstanceID string  `json:"instance_id"`
		PlayerID   int     `json:"player_id"`
		Reward     float32 `json:"reward"`
		Done       bool    `json:"done"`
	}
	
	if err := c.BindJSON(&req); err != nil {
		log.Printf("[CPU Server] Error parsing reward update request: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}
	
	s.metadataMutex.Lock()
	s.metadata.TotalReward += req.Reward
	s.metadata.RewardHistory = append(s.metadata.RewardHistory, req.Reward)
	s.metadataMutex.Unlock()
	
	log.Printf("[CPU Server] Reward updated: instance=%s, player=%d, reward=%f, done=%v, total_reward=%f",
		req.InstanceID, req.PlayerID, req.Reward, req.Done, s.metadata.TotalReward)
	
	c.JSON(http.StatusOK, gin.H{
		"status":       "success",
		"total_reward": s.metadata.TotalReward,
	})
}

func (s *TrainingServer) Run() error {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	
	// Configure Gin to use our log file
	r.Use(gin.Recovery())
	r.Use(gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		// Skip logging for /get_action requests
		if param.Path == "/get_action" {
			return ""
		}
		return fmt.Sprintf("[CPU Server] [%s] %s %s %d %s\n",
			param.TimeStamp.Format("2006-01-02 15:04:05"),
			param.Method,
			param.Path,
			param.StatusCode,
			param.ErrorMessage,
		)
	}))
	
	r.POST("/get_action", s.handleGetAction)
	r.POST("/train", s.handleTrain)
	r.POST("/save_model", s.handleSaveModel)
	r.POST("/update_reward", s.handleUpdateReward)

	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", s.port),
		Handler: r,
	}
	
	go func() {
		<-s.shutdownChan
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		server.Shutdown(ctx)
	}()
	
	log.Printf("[CPU Server] Training server starting on port %d", s.port)
	return server.ListenAndServe()
}

func findLatestModel() (string, error) {
	// Look for model files in the current directory
	files, err := os.ReadDir(".")
	if err != nil {
		return "", fmt.Errorf("failed to read directory: %v", err)
	}

	var latestModel string
	var highestCycle int = -1

	for _, file := range files {
		if !file.IsDir() && strings.HasPrefix(file.Name(), "model_") && strings.HasSuffix(file.Name(), ".json") {
			// Extract cycle number from filename (e.g., "model_42.json" -> 42)
			cycleStr := strings.TrimSuffix(strings.TrimPrefix(file.Name(), "model_"), ".json")
			cycle, err := strconv.Atoi(cycleStr)
			if err != nil {
				continue // Skip files that don't match the pattern
			}
			if cycle > highestCycle {
				highestCycle = cycle
				latestModel = file.Name()
			}
		}
	}

	if latestModel == "" {
		return "", nil // No model found
	}

	return latestModel, nil
}

func deleteAllModels() error {
	files, err := os.ReadDir(".")
	if err != nil {
		return fmt.Errorf("failed to read directory: %v", err)
	}

	for _, file := range files {
		if !file.IsDir() && strings.HasPrefix(file.Name(), "model_") && strings.HasSuffix(file.Name(), ".json") {
			if err := os.Remove(file.Name()); err != nil {
				return fmt.Errorf("failed to delete model %s: %v", file.Name(), err)
			}
			log.Printf("[CPU Server] Deleted model file: %s", file.Name())
		}
	}
	return nil
}

func main() {
	// Set up logging to file
	logFile, err := os.OpenFile(LogFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal("Failed to open log file:", err)
	}
	defer logFile.Close()
	
	// Create a custom logger that includes timestamps and only writes to file
	logger := log.New(logFile, "", log.LstdFlags)
	
	// Replace the default logger with our custom logger
	log.SetOutput(logger.Writer())
	
	// Disable Gin's default logging
	gin.SetMode(gin.ReleaseMode)
	gin.DefaultWriter = logFile
	
	// Set GOMAXPROCS
	runtime.GOMAXPROCS(NumWorkers)
	
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())
	
	// Create new server
	server := NewTrainingServer(Port)

	// Try to find and load the latest model
	latestModel, err := findLatestModel()
	if err != nil {
		log.Printf("[CPU Server] Error finding latest model: %v", err)
	} else if latestModel != "" {
		log.Printf("[CPU Server] Found latest model: %s", latestModel)
		if err := server.model.Load(latestModel); err != nil {
			log.Printf("[CPU Server] Error loading model: %v", err)
		} else {
			log.Printf("[CPU Server] Successfully loaded model from %s", latestModel)
		}
	}

	// Delete all existing models
	if err := deleteAllModels(); err != nil {
		log.Printf("[CPU Server] Error deleting models: %v", err)
	} else {
		log.Printf("[CPU Server] Successfully deleted all existing models")
	}
	
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	
	log.Printf("[CPU Server] Starting on port %d", Port)
	go func() {
		if err := server.Run(); err != nil {
			log.Printf("[CPU Server] Error: %v", err)
		}
	}()
	
	<-sigChan
	log.Println("[CPU Server] Received shutdown signal")
	close(server.shutdownChan)
	log.Println("[CPU Server] Stopped")
} 