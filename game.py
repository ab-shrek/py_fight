import pygame
import math
import time
import json
import requests
import gflags
import sys
from dataclasses import dataclass
from typing import Tuple, Optional
import logging
import threading
import os
import random

# Define gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_string('map', 'default', 'Map to use for the game')
gflags.DEFINE_integer('timeout', 120, 'Game timeout in seconds')
gflags.DEFINE_boolean('headless', True, 'Run game without graphics')
gflags.DEFINE_string('server_url', 'http://localhost:5001', 'URL of the training server')
gflags.DEFINE_boolean('use_ai', True, 'Use AI players')
gflags.DEFINE_boolean('is_first_in_cycle', False, 'Whether this is the first game in the cycle')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parallel_games.log", mode='a')  # Use the same log file as other components
    ]
)

@dataclass
class Player:
    x: float
    y: float
    angle: float
    health: int
    color: Tuple[int, int, int]
    is_ai: bool
    last_shot: float = 0
    shoot_cooldown: float = 0.5  # seconds between shots
    ROTATION_SPEED: float = 5.0  # degrees per rotation step

    def move(self, dx: float, dy: float, walls: list, other_player: Optional['Player'] = None) -> Tuple[bool, bool]:
        # Grid cell size
        CELL_SIZE = 40
        
        # Calculate current grid position
        current_grid_x = int(self.x / CELL_SIZE)
        current_grid_y = int(self.y / CELL_SIZE)
        
        # Calculate target grid position
        target_x = self.x + dx
        target_y = self.y + dy
        target_grid_x = int(target_x / CELL_SIZE)
        target_grid_y = int(target_y / CELL_SIZE)
        
        # Check if target position is valid
        can_move_x = True
        can_move_y = True
        
        # Check wall collisions
        for wall in walls:
            # Convert wall to grid coordinates
            wall_grid_left = int(wall.left / CELL_SIZE)
            wall_grid_right = int(wall.right / CELL_SIZE)
            wall_grid_top = int(wall.top / CELL_SIZE)
            wall_grid_bottom = int(wall.bottom / CELL_SIZE)
            
            # Check if target position would be inside a wall
            if (wall_grid_left <= target_grid_x <= wall_grid_right and 
                wall_grid_top <= target_grid_y <= wall_grid_bottom):
                if target_grid_x != current_grid_x:
                    can_move_x = False
                if target_grid_y != current_grid_y:
                    can_move_y = False
        
        # Check player collision
        if other_player:
            other_grid_x = int(other_player.x / CELL_SIZE)
            other_grid_y = int(other_player.y / CELL_SIZE)
            
            if target_grid_x == other_grid_x and target_grid_y == other_grid_y:
                if target_grid_x != current_grid_x:
                    can_move_x = False
                if target_grid_y != current_grid_y:
                    can_move_y = False
        
        # Apply movement if valid
        if can_move_x:
            self.x = target_x
        if can_move_y:
            self.y = target_y
        
        # Keep within game boundaries
        self.x = max(20, min(780, self.x))
        self.y = max(20, min(580, self.y))
        
        return can_move_x, can_move_y

    def rotate(self, direction: float) -> None:
        """
        Rotate the player by the specified amount.
        Positive values rotate clockwise, negative values rotate anticlockwise.
        Args:
            direction: Rotation amount in degrees. Positive for clockwise, negative for anticlockwise.
        """
        self.angle = (self.angle + direction * self.ROTATION_SPEED) % 360

    def shoot(self, current_time: float) -> bool:
        if current_time - self.last_shot >= self.shoot_cooldown:
            self.last_shot = current_time
            return True
        return False

    def take_damage(self, amount: int) -> None:
        self.health = max(0, self.health - amount)

@dataclass
class Bullet:
    x: float
    y: float
    angle: float
    start_time: float
    shooter_idx: int

class Game:
    # Static counter for game instances
    _instance_counter = 0
    _cycle_size = 10  # Number of games in each cycle
    
    def __init__(self):
        # Initialize pygame if not headless
        if not FLAGS.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("2 Player Battle Game")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        # Load map first so we can check wall collisions
        self.walls = self.load_map(FLAGS.map)
        
        # Set cycle information from command line flag
        self.is_first_in_cycle = FLAGS.is_first_in_cycle
        
        # Generate random positions for players
        def get_random_position():
            while True:
                # Keep away from walls (at least 2 grid cells)
                x = random.randint(80, 720)  # 2 * CELL_SIZE from walls
                y = random.randint(80, 520)  # 2 * CELL_SIZE from walls
                
                # Check if position is valid (not inside any wall or obstacle)
                valid = True
                for wall in self.walls:
                    # Convert wall to grid coordinates
                    wall_grid_left = int(wall.left / 40)
                    wall_grid_right = int(wall.right / 40)
                    wall_grid_top = int(wall.top / 40)
                    wall_grid_bottom = int(wall.bottom / 40)
                    
                    # Convert player position to grid coordinates
                    player_grid_x = int(x / 40)
                    player_grid_y = int(y / 40)
                    
                    # Check if player is too close to wall (within 2 grid cells)
                    if (wall_grid_left - 2 <= player_grid_x <= wall_grid_right + 2 and 
                        wall_grid_top - 2 <= player_grid_y <= wall_grid_bottom + 2):
                        valid = False
                        break
                
                if valid:
                    return x, y
        
        # Get random positions for both players
        pos1 = get_random_position()
        pos2 = get_random_position()
        
        # Make sure players are not too close to each other (at least 5 grid cells)
        while math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) < 200:  # 5 * CELL_SIZE
            pos2 = get_random_position()
        
        # Game state
        self.players = [
            Player(pos1[0], pos1[1], random.randint(0, 359), 100, 
                  (0, 255, 0) if self.is_first_in_cycle else (255, 0, 0),  # Green for prediction, Red for random
                  FLAGS.use_ai),  # Player 1
            Player(pos2[0], pos2[1], random.randint(0, 359), 100, 
                  (255, 255, 0) if self.is_first_in_cycle else (0, 0, 255),  # Yellow for prediction, Blue for random
                  FLAGS.use_ai)   # Player 2
        ]
        
        # Game timing
        self.start_time = time.time()
        self.game_over = False
        self.winner = None

        # Create unique instance ID for this game
        self.instance_id = f"game_{int(time.time())}_{os.getpid()}"
        
        # Experience file for this instance
        self.experience_file = f"{self.instance_id}.jsonl"
        
        # Active bullets
        self.bullets = []
        
        # Hit tracking for rewards
        self.player_hits = [False, False]  # Track if each player hit the other
        self.player_hit_by = [False, False]  # Track if each player was hit
        
        logging.info(f"Game started with instance ID: {self.instance_id}")
        logging.info(f"Player 1 spawned at ({int(pos1[0])}, {int(pos1[1])})")
        logging.info(f"Player 2 spawned at ({int(pos2[0])}, {int(pos2[1])})")
        logging.info(f"Using {'model' if self.is_first_in_cycle else 'random'} actions")

    def load_map(self, map_name: str) -> list:
        """Load map with random obstacles"""
        # Fixed outer walls
        walls = [
            pygame.Rect(0, 0, 800, 20),  # Top wall
            pygame.Rect(0, 580, 800, 20),  # Bottom wall
            pygame.Rect(0, 0, 20, 600),  # Left wall
            pygame.Rect(780, 0, 20, 600),  # Right wall
        ]
        
        # Add two random obstacles
        for _ in range(2):
            # Random position (keeping away from edges and other obstacles)
            x = random.randint(100, 700)
            y = random.randint(100, 500)
            
            # Random size (between 100 and 200 pixels)
            width = random.randint(100, 200)
            height = random.randint(20, 40)
            
            # Random angle
            angle = random.randint(0, 359)
            
            # Create rotated rectangle
            rect = pygame.Rect(x, y, width, height)
            
            # Check if this obstacle overlaps with existing ones
            overlaps = False
            for existing_wall in walls:
                if rect.colliderect(existing_wall):
                    overlaps = True
                    break
            
            # If no overlap, add the obstacle
            if not overlaps:
                walls.append(rect)
                logging.info(f"Added obstacle at ({x}, {y}) with size {width}x{height} and angle {angle}°")
        
        return walls

    def get_observation(self, player_idx: int) -> list:
        """Get observation for AI player"""
        player = self.players[player_idx]
        opponent = self.players[1 - player_idx]
        
        # Calculate relative position and angle to opponent
        dx = opponent.x - player.x
        dy = opponent.y - player.y
        distance = math.sqrt(dx*dx + dy*dy)
        angle_to_opponent = math.degrees(math.atan2(dy, dx)) - player.angle
        
        # Normalize angle to [-180, 180]
        angle_to_opponent = (angle_to_opponent + 180) % 360 - 180
        
        # Create 20x15 grid centered on player
        grid_size_x = 20  # cells
        grid_size_y = 15  # cells
        cell_size = 40    # pixels per cell
        grid = []
        
        # Calculate grid boundaries
        grid_left = player.x - (grid_size_x * cell_size) / 2
        grid_top = player.y - (grid_size_y * cell_size) / 2
        
        # Fill grid with wall information
        for y in range(grid_size_y):
            for x in range(grid_size_x):
                cell_x = grid_left + x * cell_size
                cell_y = grid_top + y * cell_size
                
                # Check if any wall intersects with this cell
                has_wall = False
                for wall in self.walls:
                    if (wall.left <= cell_x + cell_size and 
                        wall.right >= cell_x and 
                        wall.top <= cell_y + cell_size and 
                        wall.bottom >= cell_y):
                        has_wall = True
                        break
                
                grid.append(1.0 if has_wall else 0.0)
        
        return [
            dx / 800.0,  # Normalized x distance to opponent
            dy / 600.0,  # Normalized y distance to opponent
            angle_to_opponent / 180.0,  # Normalized angle to opponent
            player.angle / 360.0,  # Normalized player's own orientation
            opponent.angle / 360.0,  # Normalized opponent's orientation
            player.health / 100.0,  # Normalized health
            opponent.health / 100.0,  # Normalized opponent health
            (time.time() - self.start_time) / FLAGS.timeout,  # Normalized time
        ] + grid  # Add grid information

    def get_ai_action(self, player_idx: int) -> Tuple[float, float, float, bool]:
        """Get action from AI server or random action based on cycle position"""
        if not self.is_first_in_cycle:
            # Generate random actions
            move_x = random.uniform(-1.0, 1.0)
            move_z = random.uniform(-1.0, 1.0)
            rotate = random.uniform(-1.0, 1.0)
            shoot = random.random()  # Random value between 0 and 1
            
            # Log the random action values
            # mario commented logging for random action
            # logging.info(f"Player {player_idx + 1} random action: move_x={move_x:.2f}, move_z={move_z:.2f}, rotate={rotate:.2f}, shoot={shoot:.2f}")
            
            return move_x, move_z, rotate, shoot > 0.5  # Convert shoot to boolean
        else:
            # Get action from AI server
            try:
                response = requests.post(
                    f"{FLAGS.server_url}/get_action",
                    json={
                        "observation": self.get_observation(player_idx),
                        "instance_id": f"player_{player_idx}",
                        "player_id": player_idx
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    action = data.get("action", [0, 0, 0])
                    # Handle both 3-value and 4-value responses
                    if len(action) == 3:
                        # Add default shoot value (0.0) for 3-value responses
                        action = tuple(action) + (0.0,)
                    elif len(action) == 4:
                        action = tuple(action)
                    else:
                        print(f"Unexpected action format: {action}")
                        action = (0, 0, 0, False)
                    
                    # Log the action values
                    move_x, move_z, rotate, shoot = action
                    # mario commented logging for AI action
                    # logging.info(f"Player {player_idx + 1} AI action: move_x={move_x:.2f}, move_z={move_z:.2f}, rotate={rotate:.2f}, shoot={shoot:.2f}")
                    return action
                else:
                    print(f"Error getting AI action: {response.status_code}")
                    return 0, 0, 0, False
            except Exception as e:
                print(f"Error connecting to AI server: {e}")
                return 0, 0, 0, False

    def add_experience(self, observation, action, reward, next_observation, done, player_idx: int):
        """Add experience to the instance-specific file"""
        # Calculate reward based on game state
        current_reward = 0
        reward_components = []
        
        # Reward for hitting opponent
        if self.player_hits[player_idx]:
            current_reward += 10.0  # Reward for hitting opponent
            self.player_hits[player_idx] = False  # Reset hit flag
            reward_components.append("hit_opponent:+10.0")
            logging.info(f"Player {player_idx + 1} received +10.0 reward for hitting opponent")
        
        # Penalty for being hit
        if self.player_hit_by[player_idx]:
            current_reward -= 9.0  # Penalty for being hit
            self.player_hit_by[player_idx] = False  # Reset hit flag
            reward_components.append("was_hit:-9.0")
            logging.info(f"Player {player_idx + 1} received -9.0 penalty for being hit")
        
        # Reward for winning
        if done and self.winner is not None:
            if self.winner == player_idx:
                current_reward += 50.0  # Big reward for winning
                reward_components.append("win:+50.0")
                logging.info(f"Player {player_idx + 1} received +50.0 reward for winning")
        
        # Log the complete reward breakdown
        # mario commented logging for reward breakdown
        if reward_components:
            logging.info(f"Player {player_idx + 1} reward breakdown: {', '.join(reward_components)} = {current_reward}")
        
        # Convert action to proper format
        move_x, move_z, rotate, shoot = action
        action_formatted = {
            "move_x": float(move_x),
            "move_z": float(move_z),
            "rotate": float(rotate),
            "shoot": bool(shoot)
        }
        
        experience = {
            "observation": observation,
            "action": action_formatted,
            "reward": current_reward,
            "next_observation": next_observation,
            "done": done,
            "timestamp": time.time()
        }
        
        try:
            with open(self.experience_file, 'a') as f:
                f.write(json.dumps(experience) + '\n')
        except Exception as e:
            logging.error(f"Error writing experience to file: {e}")

    def cleanup(self):
        """Clean up experience file when game ends"""
        try:
            if os.path.exists(self.experience_file):
                os.remove(self.experience_file)
                logging.info(f"Cleaned up experience file: {self.experience_file}")
        except Exception as e:
            logging.error(f"Error cleaning up experience file: {e}")

    def handle_input(self, player_idx: int) -> None:
        """Handle player input or AI action"""
        player = self.players[player_idx]
        
        if player.is_ai:
            # Get AI action
            move_x, move_z, rotate, shoot = self.get_ai_action(player_idx)
            
            # Apply movement
            self.last_move_success = player.move(move_x * 5, move_z * 5, self.walls, self.players[1 - player_idx])
            # AI rotation: positive values rotate clockwise, negative rotate anticlockwise
            player.rotate(rotate)
            
            # Handle shooting
            if shoot > 0.5 and player.shoot(time.time()):
                self.handle_shot(player_idx)
        else:
            # Handle keyboard input for human player
            keys = pygame.key.get_pressed()
            
            # Movement
            if keys[pygame.K_w]:
                self.last_move_success = player.move(0, -5, self.walls, self.players[1 - player_idx])
            if keys[pygame.K_s]:
                self.last_move_success = player.move(0, 5, self.walls, self.players[1 - player_idx])
            if keys[pygame.K_a]:
                self.last_move_success = player.move(-5, 0, self.walls, self.players[1 - player_idx])
            if keys[pygame.K_d]:
                self.last_move_success = player.move(5, 0, self.walls, self.players[1 - player_idx])
            
            # Rotation - using arrow keys
            if keys[pygame.K_LEFT]:
                player.rotate(-1)  # Rotate anticlockwise
            if keys[pygame.K_RIGHT]:
                player.rotate(1)   # Rotate clockwise
            
            # Shooting
            if keys[pygame.K_SPACE] and player.shoot(time.time()):
                self.handle_shot(player_idx)

        # Add experience to buffer after each step
        self.add_experience(
            observation=self.get_observation(player_idx),
            action=[move_x, move_z, rotate, shoot] if player.is_ai else [0, 0, 0, 0],  # Simplified for human players
            reward=0,  # Calculate reward based on game state
            next_observation=self.get_observation(player_idx),
            done=self.game_over,
            player_idx=player_idx
        )

    def handle_shot(self, shooter_idx: int) -> None:
        """Handle shooting logic"""
        shooter = self.players[shooter_idx]
        target = self.players[1 - shooter_idx]
        
        # Create new bullet
        bullet = Bullet(
            x=shooter.x,
            y=shooter.y,
            angle=shooter.angle,
            start_time=time.time(),
            shooter_idx=shooter_idx
        )
        self.bullets.append(bullet)

        # mario commented logging for bullet firing
        # logging.info(f"Player {shooter_idx + 1} fired a bullet from position ({int(shooter.x)}, {int(shooter.y)}) at angle {int(shooter.angle)}°")
        
        # Calculate shot trajectory
        shot_x = shooter.x
        shot_y = shooter.y
        angle_rad = math.radians(shooter.angle)
        
        # Check if shot hits target or wall
        while True:
            shot_x += math.cos(angle_rad) * 5
            shot_y += math.sin(angle_rad) * 5
            
            # Check if shot is out of bounds
            if shot_x < 0 or shot_x > 800 or shot_y < 0 or shot_y > 600:
                # logging.info(f"Player {shooter_idx + 1}'s bullet went out of bounds at ({int(shot_x)}, {int(shot_y)})")
                return
            
            # Check wall collision
            for wall in self.walls:
                if wall.collidepoint(shot_x, shot_y):
                    # mario commented logging for bullet hitting a wall
                    # logging.info(f"Player {shooter_idx + 1}'s bullet hit a wall at ({int(shot_x)}, {int(shot_y)})")
                    return
            
            # Check target hit
            if (abs(shot_x - target.x) < 20 and 
                abs(shot_y - target.y) < 20):
                target.take_damage(10)
                # Set hit flags for reward calculation
                self.player_hits[shooter_idx] = True  # Shooter hit target
                self.player_hit_by[1 - shooter_idx] = True  # Target was hit
                logging.info(f"Player {shooter_idx + 1} hit Player {1 - shooter_idx + 1}! Target health: {target.health}")
                return

    def check_game_over(self) -> bool:
        """Check if game is over"""
        current_time = time.time()
        
        # Check timeout
        if current_time - self.start_time >= FLAGS.timeout:
            self.game_over = True
            # Determine winner based on health
            if self.players[0].health > self.players[1].health:
                self.winner = 0
            elif self.players[1].health > self.players[0].health:
                self.winner = 1
            else:
                self.winner = -1  # Draw
            logging.info(f"Game over due to timeout. Winner: Player {self.winner + 1}")
            return True
        
        # Check health
        for i, player in enumerate(self.players):
            if player.health <= 0:
                self.game_over = True
                self.winner = 1 - i
                logging.info(f"Game over due to health. Winner: Player {self.winner + 1}")
                return True
        
        return False

    def draw(self) -> None:
        """Draw game state"""
        if FLAGS.headless:
            return
            
        self.screen.fill((0, 0, 0))
        
        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, (100, 100, 100), wall)
        
        # Draw players
        for player in self.players:
            # Draw player body
            pygame.draw.circle(self.screen, player.color, (int(player.x), int(player.y)), 20)
            
            # Draw direction indicator
            end_x = player.x + math.cos(math.radians(player.angle)) * 30
            end_y = player.y + math.sin(math.radians(player.angle)) * 30
            pygame.draw.line(self.screen, (255, 255, 255), 
                           (player.x, player.y), (end_x, end_y), 2)
            
            # Draw health bar
            health_width = 40 * (player.health / 100)
            pygame.draw.rect(self.screen, (0, 255, 0),
                           (player.x - 20, player.y - 30, health_width, 5))
        
        # Draw bullets
        current_time = time.time()
        active_bullets = []
        for bullet in self.bullets:
            if current_time - bullet.start_time < 0.5:  # Show bullet for 0.5 seconds
                # Calculate bullet position based on time
                distance = (current_time - bullet.start_time) * 500  # 500 pixels per second
                bullet_x = bullet.x + math.cos(math.radians(bullet.angle)) * distance
                bullet_y = bullet.y + math.sin(math.radians(bullet.angle)) * distance
                
                # Calculate full trajectory until hit
                angle_rad = math.radians(bullet.angle)
                shot_x = bullet.x
                shot_y = bullet.y
                hit_point = None
                
                # Check trajectory until hit
                while True:
                    shot_x += math.cos(angle_rad) * 5
                    shot_y += math.sin(angle_rad) * 5
                    
                    # Check if shot is out of bounds
                    if shot_x < 0 or shot_x > 800 or shot_y < 0 or shot_y > 600:
                        hit_point = (shot_x, shot_y)
                        break
                    
                    # Check wall collision
                    for wall in self.walls:
                        if wall.collidepoint(shot_x, shot_y):
                            hit_point = (shot_x, shot_y)
                            break
                    
                    # Check target hit
                    target = self.players[1 - bullet.shooter_idx]
                    if (abs(shot_x - target.x) < 20 and 
                        abs(shot_y - target.y) < 20):
                        hit_point = (shot_x, shot_y)
                        break
                    
                    if hit_point:
                        break
                
                if hit_point:
                    # Draw dotted trail from start to hit point
                    total_distance = math.sqrt((hit_point[0] - bullet.x)**2 + (hit_point[1] - bullet.y)**2)
                    num_dots = int(total_distance / 10)  # One dot every 10 pixels
                    
                    for i in range(num_dots):
                        t = i / num_dots
                        dot_x = bullet.x + (hit_point[0] - bullet.x) * t
                        dot_y = bullet.y + (hit_point[1] - bullet.y) * t
                        pygame.draw.circle(self.screen, (255, 255, 0), (int(dot_x), int(dot_y)), 2)
                    
                    # Draw bullet at current position
                    pygame.draw.circle(self.screen, (255, 255, 0), 
                                     (int(bullet_x), int(bullet_y)), 3)
                
                active_bullets.append(bullet)
        
        # Update active bullets
        self.bullets = active_bullets
        
        # Draw game over screen
        if self.game_over:
            if self.winner == -1:
                text = "Game Over - Draw!"
            else:
                text = f"Game Over - Player {self.winner + 1} Wins!"
            text_surface = self.font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(400, 300))
            self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()

    def run(self) -> None:
        """Main game loop"""
        running = True
        try:
            while running and not self.game_over:
                if not FLAGS.headless:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                
                # Handle player input
                self.handle_input(0)
                self.handle_input(1)
                
                # Check game over
                self.check_game_over()
                
                # Draw game state
                self.draw()
                
                if not FLAGS.headless:
                    self.clock.tick(60)
        finally:
            if not FLAGS.headless:
                pygame.quit()

def main():
    try:
        argv = FLAGS(sys.argv)
    except gflags.FlagsError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    game = Game()
    game.run()

if __name__ == "__main__":
    main() 