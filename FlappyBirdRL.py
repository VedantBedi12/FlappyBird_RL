import pygame, sys, random 
import torch
import random
import numpy as np

class FlappyBirdGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((576, 900))
        self.clock = pygame.time.Clock()
        self.gravity = 0.25
        self.bird_movement = 0
        self.game_active = True
        self.score = 0
        self.high_score = 0
        self.reward = 0

        # Load assets
        self.bg_surface = pygame.transform.scale2x(pygame.image.load('assets/background-day.png').convert())
        self.floor_surface = pygame.transform.scale2x(pygame.image.load('assets/base.png').convert())
        self.bird_frames = [
            pygame.transform.scale2x(pygame.image.load('assets/bluebird-downflap.png').convert_alpha()),
            pygame.transform.scale2x(pygame.image.load('assets/bluebird-midflap.png').convert_alpha()),
            pygame.transform.scale2x(pygame.image.load('assets/bluebird-upflap.png').convert_alpha())
        ]
        self.pipe_surface = pygame.transform.scale2x(pygame.image.load('assets/pipe-green.png'))
        self.pipe_height = [400, 500, 600]

        # Initialize game state
        self.bird_index = 0
        self.bird_surface = self.bird_frames[self.bird_index]
        self.bird_rect = self.bird_surface.get_rect(center=(100, 512))
        self.pipe_list = []
        self.floor_x_pos = 0
        

        # Sounds
        self.flap_sound = pygame.mixer.Sound('sound/sfx_wing.wav')
        self.score_sound = pygame.mixer.Sound('sound/sfx_point.wav')

        # Timers
        self.SPAWNPIPE = pygame.USEREVENT
        pygame.time.set_timer(self.SPAWNPIPE, 1200)

    def reset(self):
        """Reset the game state."""
        self.bird_rect.center = (100, 512)
        self.bird_movement = 0
        self.pipe_list.clear()
        self.score = 0
        self.game_active = True
        # print(self.reward)
        self.reward = 0.1
        

    def play_step(self, action):
        """
        Take a step in the game.
        :param action: [0, 1] where 1 means "flap".
        :return: reward, done, score
        """
        a =[0,0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == self.SPAWNPIPE:
                a = self.create_pipe()
                self.pipe_list.extend(a)
                # self.pipe_list.extend(self.create_pipe())

        # Initialize reward
        self.reward += 0.1 # Increased reward for surviving

        # Bird movement
        if action[1] == 1:  # Flap
            self.reward -= 0.1  # Reduced penalty for flapping
            self.bird_movement = -3
            self.flap_sound.play()

        self.bird_movement += self.gravity
        self.bird_rect.centery += self.bird_movement

        # Penalize hitting the top or bottom of the screen
        if self.bird_rect.top <= -100 or self.bird_rect.bottom >= 800:
            self.reward -= 15  # High penalty for going out of bounds
            self.game_active = False 
            return self.reward, True, self.score
        
        # Reward for staying in the middle of the game
        # screen_center = self.screen.get_height() / 2
        # distance_from_center = abs(self.bird_rect.centery - screen_center)
        # reward += max(0, 10 - distance_from_center / 10)  # Reward decreases as bird moves away from center

        
        for pipe in self.pipe_list:
            if self.bird_rect.centerx == pipe.centerx:
                self.reward += 7 # Positive reward for being in line with a pipe
                if a[0] !=0 and a[1] != 0:
                    if self.bird_rect.centery < a[0].centery and self.bird_rect.centery > a[1].centery:
                        self.reward += 10  # Positive reward for passing a pipe

        # Check collisions with pipes
        if not self.check_collision():
            self.game_active = False
            self.reward -= 10  # Negative reward for dying
            return self.reward, True, self.score

        # Move pipes and check for passing
        previous_score = int(self.score)  # Keep track of score before moving pipes
        # self.pipe_list = self.move_pipes(self.pipe_list)
        self.pipe_list = self.move_pipes(self.pipe_list)

        # Reward for passing pipes
        if int(self.score) > previous_score:
            self.reward += 20  # Reduced reward for passing a pipe

        # Update score and draw everything
        self.score += 0.01
        self.draw_elements()
        self.clock.tick(60)



        return self.reward, False, self.score

    def get_game_state(self):
        """
        Extract the current game state as a feature vector.
        :return: [bird_y, bird_vel, pipe_dist, top_pipe, bottom_pipe, gap_center, bird_gap_dist, score]
        """
        bird_y = self.bird_rect.centery
        bird_vel = self.bird_movement

        if self.pipe_list:
            nearest_pipe = self.pipe_list[0]
            for pipe in self.pipe_list:
                if pipe.centerx > self.bird_rect.centerx:
                    nearest_pipe = pipe
                    break

            pipe_dist = nearest_pipe.centerx - self.bird_rect.centerx
            if nearest_pipe.bottom >= 900:
                top_pipe = nearest_pipe.top
                bottom_pipe = nearest_pipe.bottom
            else:
                top_pipe = nearest_pipe.bottom
                bottom_pipe = nearest_pipe.top
            gap_center = (top_pipe + bottom_pipe) / 2
            bird_gap_dist = bird_y - gap_center
        else:
            pipe_dist, top_pipe, bottom_pipe, gap_center, bird_gap_dist = 0, 0, 0, 0, 0
        # Distance to the nearest pipe
        nearest_pipe_dist = min([pipe.centerx - self.bird_rect.centerx for pipe in self.pipe_list if pipe.centerx > self.bird_rect.centerx], default=0)

        # Bird's distance from the top and bottom of the screen
        bird_top_dist = self.bird_rect.top
        bird_bottom_dist = self.screen.get_height() - self.bird_rect.bottom

        # return [bird_y, bird_vel, pipe_dist, top_pipe, bottom_pipe, gap_center, bird_gap_dist, self.score]

        return [bird_y, bird_vel, pipe_dist, top_pipe, bottom_pipe, gap_center, bird_gap_dist, self.score, nearest_pipe_dist, bird_top_dist, bird_bottom_dist]

    def check_collision(self):
        """Check for collisions with pipes or boundaries."""
        for pipe in self.pipe_list:
            if self.bird_rect.colliderect(pipe):
                return False
        if self.bird_rect.top <= -100 or self.bird_rect.bottom >= 900:
            return False
        return True

    def create_pipe(self):
        """Create a new pair of pipes."""
        random_pipe_pos = random.choice(self.pipe_height)
        bottom_pipe = self.pipe_surface.get_rect(midtop=(700, random_pipe_pos))
        top_pipe = self.pipe_surface.get_rect(midbottom=(700, random_pipe_pos - 300))
        return bottom_pipe, top_pipe

    def move_pipes(self, pipes):
        """Move pipes to the left."""
        for pipe in pipes:
            pipe.centerx -= 5
        return [pipe for pipe in pipes if pipe.centerx > -50]


    def draw_elements(self):
        """Draw all elements on the screen."""
        self.screen.blit(self.bg_surface, (0, 0))
        for pipe in self.pipe_list:
            if pipe.bottom >= 900:
                self.screen.blit(self.pipe_surface, pipe)
            else:
                flip_pipe = pygame.transform.flip(self.pipe_surface, False, True)
                self.screen.blit(flip_pipe, pipe)
        self.screen.blit(self.floor_surface, (self.floor_x_pos, 800))
        rotated_bird = pygame.transform.rotozoom(self.bird_surface, -self.bird_movement * 3, 1)
        self.screen.blit(rotated_bird, self.bird_rect)
        pygame.display.update()