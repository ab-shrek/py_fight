# PyFight Game

A 2D combat game where AI agents learn to fight each other using reinforcement learning.

## Game Mechanics

### Movement System
- Grid-based movement (40x40 pixel cells)
- Players can move in 8 directions
- Collision detection with walls and other players
- Players spawn at least 2 grid cells away from walls
- Minimum 5 grid cells separation between players

### Combat System
- Players can shoot in the direction they're facing
- Shooting has a 0.5 second cooldown
- Bullets travel in straight lines
- Players have 100 health points
- Each hit deals 10 damage

### AI Training
- Uses random actions for exploration
- Collects experiences for model training
- Simplified reward structure focused on combat

## Reward System

### Positive Rewards
- Hitting opponent: +10.0
- Winning the game: +50.0

### Negative Rewards
- Getting hit: -9.0

## Game Features
- 800x600 pixel game area
- Wall obstacles for tactical gameplay
- Health bars for both players
- Visual bullet trails
- Game timeout after 120 seconds

## Training
- Experiences are saved to JSONL files
- Each game instance has a unique ID
- Random actions for maximum exploration
- Focus on combat-based learning

## Controls
- Player 1: WASD for movement, Arrow keys for rotation, Space to shoot
- Player 2: AI-controlled

## Running the Game
```bash
python game.py
```

Optional flags:
- `--map`: Map to use (default: 'default')
- `--timeout`: Game timeout in seconds (default: 120)
- `--headless`: Run without graphics (default: True)
- `--server_url`: Training server URL (default: 'http://localhost:5001')
- `--use_ai`: Use AI players (default: True)

## Command Line Options
- `--map`: Select map to use (default: 'default')
- `--timeout`: Game timeout in seconds (default: 120)
- `--headless`: Run game without graphics (default: True)
- `--server_url`: URL of the training server (default: 'http://localhost:5001')
- `--use_ai`: Use AI players (default: True)

## Development
The game uses:
- Pygame for graphics and input handling
- Go for the training server
- CUDA for GPU acceleration (when using GPU server)

## Model Architecture

### Input Space
The AI model receives a rich observation space including:
- Relative position to opponent (normalized x, y coordinates)
- Angle to opponent (normalized to [-1, 1])
- Player health (normalized to [0, 1])
- Opponent health (normalized to [0, 1])
- Game time (normalized to [0, 1])
- 20x15 grid of wall information (300 binary values)

### Action Space
The model outputs a 4-dimensional continuous action space:
- `move_x`: Horizontal movement (-1 to 1)
- `move_z`: Vertical movement (-1 to 1)
- `rotate`: Rotation direction and speed (-1 to 1)
  - Positive values: Rotate clockwise
  - Negative values: Rotate anticlockwise
- `shoot`: Shooting probability (0 to 1)

### Reward Structure
The model is trained with a comprehensive reward system:

#### Positive Rewards
- +10.0 for hitting opponent
- +50.0 for winning
- +20.0 bonus for winning by shooting
- +2.0 for shooting attempts
- +3.0 for shooting when well-aligned (within 30 degrees)
- +2.0 for shooting at close range (< 200 pixels)
- +1.0 for moving towards opponent's grid cell
- +1.0 for rotating towards opponent
- +0.5 for rotation attempts

#### Negative Rewards (Penalties)
- -9.0 for being hit by opponent
- -2.0 for getting stuck (attempting to move but blocked)
- -1.0 for staying in same grid cell as opponent without shooting
- -1.0 for staying in corner grid cells
- -1.0 for not shooting when well-aligned and close
- -0.5 for rotating away from opponent

The penalty system encourages:
- Avoiding grid cell collisions with opponent
- Staying away from map corners
- Strategic grid-based positioning
- Taking advantage of good shooting opportunities
- Efficient rotation and movement

### Training Process
- Experience collection from parallel game instances
- GPU-accelerated training on the server
- Real-time model updates
- Experience replay with prioritized sampling
- Continuous learning from game outcomes 