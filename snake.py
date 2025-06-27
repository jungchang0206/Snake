import pygame
import random
import numpy as np
from collections import deque
import time

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Game parameters
BLOCK_SIZE = 20
SPEED = 10 

class SnakeGame:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        # Initialize game state
        self.direction = random.choice([pygame.K_RIGHT, pygame.K_LEFT, pygame.K_UP, pygame.K_DOWN])
        self.head = [self.width//2, self.height//2]
        self.snake = [self.head.copy(), 
                      [self.head[0]-BLOCK_SIZE, self.head[1]], 
                      [self.head[0]-(2*BLOCK_SIZE), self.head[1]]]
        self.score = 0
        self.food = self._place_food()
        self.frame_iteration = 0
        return self._get_state()
    
    def _place_food(self):
        while True:
            food = [random.randrange(0, self.width-BLOCK_SIZE, BLOCK_SIZE),
                    random.randrange(0, self.height-BLOCK_SIZE, BLOCK_SIZE)]
            if food not in self.snake:
                return food
    
    def _get_state(self):
        # Get the current state representation for the AI
        head = self.snake[0]
        
        # Directions: [straight, right, left] relative to current direction
        dir_l = self.direction == pygame.K_LEFT
        dir_r = self.direction == pygame.K_RIGHT
        dir_u = self.direction == pygame.K_UP
        dir_d = self.direction == pygame.K_DOWN
        
        # Check danger in 3 directions (straight, right, left)
        danger_straight = danger_right = danger_left = 0
        
        # Calculate new positions
        if dir_l:
            new_head_straight = [head[0] - BLOCK_SIZE, head[1]]
            new_head_right = [head[0], head[1] - BLOCK_SIZE]
            new_head_left = [head[0], head[1] + BLOCK_SIZE]
        elif dir_r:
            new_head_straight = [head[0] + BLOCK_SIZE, head[1]]
            new_head_right = [head[0], head[1] + BLOCK_SIZE]
            new_head_left = [head[0], head[1] - BLOCK_SIZE]
        elif dir_u:
            new_head_straight = [head[0], head[1] - BLOCK_SIZE]
            new_head_right = [head[0] + BLOCK_SIZE, head[1]]
            new_head_left = [head[0] - BLOCK_SIZE, head[1]]
        elif dir_d:
            new_head_straight = [head[0], head[1] + BLOCK_SIZE]
            new_head_right = [head[0] - BLOCK_SIZE, head[1]]
            new_head_left = [head[0] + BLOCK_SIZE, head[1]]
        
        # Check for collisions
        danger_straight = self._is_collision(new_head_straight)
        danger_right = self._is_collision(new_head_right)
        danger_left = self._is_collision(new_head_left)
        
        # Food location
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        
        return np.array([
            # Danger directions
            danger_straight, danger_right, danger_left,
            
            # Current direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Food location
            food_left, food_right, food_up, food_down
        ], dtype=int)
    
    def _is_collision(self, point=None):
        if point is None:
            point = self.snake[0]
        # Hits boundary
        if (point[0] >= self.width or point[0] < 0 or 
            point[1] >= self.height or point[1] < 0):
            return True
        # Hits itself
        if point in self.snake[1:]:
            return True
        return False
    
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input (for human play)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move (action is [straight, right, left])
        self._move(action)
        
        # 3. Check if game over
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 4. Place new food or just move
        if self.snake[0] == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()
        
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. Return game over and score
        return reward, game_over, self.score
    
    def _move(self, action):
        # Action: [straight, right, left]
        
        clock_wise = [pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT, pygame.K_UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):  # Straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # Right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # Left turn [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        
        x = self.snake[0][0]
        y = self.snake[0][1]
        if self.direction == pygame.K_RIGHT:
            x += BLOCK_SIZE
        elif self.direction == pygame.K_LEFT:
            x -= BLOCK_SIZE
        elif self.direction == pygame.K_DOWN:
            y += BLOCK_SIZE
        elif self.direction == pygame.K_UP:
            y -= BLOCK_SIZE
            
        self.snake.insert(0, [x, y])
    
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt[0]+4, pt[1]+4, BLOCK_SIZE-8, BLOCK_SIZE-8))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        
        font = pygame.font.SysFont('arial', 20)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, [0, 0])
        pygame.display.flip()

if __name__ == '__main__':
    game = SnakeGame()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                # Prevent the snake from reversing directly
                if event.key == pygame.K_RIGHT and game.direction != pygame.K_LEFT:
                    game.direction = pygame.K_RIGHT
                elif event.key == pygame.K_LEFT and game.direction != pygame.K_RIGHT:
                    game.direction = pygame.K_LEFT
                elif event.key == pygame.K_UP and game.direction != pygame.K_DOWN:
                    game.direction = pygame.K_UP
                elif event.key == pygame.K_DOWN and game.direction != pygame.K_UP:
                    game.direction = pygame.K_DOWN

        # Only call play_step, which handles movement and growth
        reward, game_over, score = game.play_step([1, 0, 0])
        if game_over:
            print('Final Score:', score)
            break
    pygame.quit()