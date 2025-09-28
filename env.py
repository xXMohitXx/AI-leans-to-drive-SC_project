import pygame
import numpy as np
import math
import random

SCREEN_SIZE = (800, 600)
ROAD_RADIUS = 40
MAX_SPEED = 8.0
MAX_STEPS_WITHOUT_IMPROVEMENT = 100  # kill if stuck


def catmull_rom_chain(points, count=300):
    """Generate smooth closed track points with Catmull-Rom spline."""
    P = points[:]
    P = P[-2:] + P + P[:2]
    curve = []
    for i in range(2, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, count // len(points)):
            t2, t3 = t * t, t * t * t
            f1 = -0.5*t3 + t2 - 0.5*t
            f2 = 1.5*t3 - 2.5*t2 + 1
            f3 = -1.5*t3 + 2*t2 + 0.5*t
            f4 = 0.5*t3 - 0.5*t2
            x = p0[0]*f1 + p1[0]*f2 + p2[0]*f3 + p3[0]*f4
            y = p0[1]*f1 + p1[1]*f2 + p2[1]*f3 + p3[1]*f4
            curve.append((x, y))
    return curve


class CarEnv:
    def __init__(self, render=False, color=None):
        self.render_mode = render
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode(SCREEN_SIZE)

        # Control points for track shape
        pts = [
            (120, 520), (250, 560), (500, 540), (700, 450),
            (740, 300), (700, 160), (500, 80), (250, 120),
            (120, 250)
        ]
        self.centerline = catmull_rom_chain(pts, count=600)

        # Build track surface
        self.track_surface = pygame.Surface(SCREEN_SIZE)
        self.track_surface.fill((30, 120, 30))
        self.road_color = (60, 60, 60)
        for (x, y) in self.centerline:
            pygame.draw.circle(self.track_surface, self.road_color, (int(x), int(y)), ROAD_RADIUS)

        self.track_mask = pygame.mask.from_threshold(self.track_surface, self.road_color, (40, 40, 40))

        # Gates (checkpoints as lines)
        self.gates = self._make_gates(12)

        # Car sprite
        self.base_car_surface = pygame.Surface((18, 36), pygame.SRCALPHA)

        self.reset()
        self.prev_pos = (self.x, self.y)

    def _make_gates(self, n):
        """Make gates perpendicular to track direction."""
        step = len(self.centerline) // n
        gates = []
        for i in range(0, len(self.centerline), step):
            cx, cy = self.centerline[i]
            nx, ny = self.centerline[(i+5) % len(self.centerline)]
            dx, dy = nx - cx, ny - cy
            length = math.hypot(dx, dy) or 1.0
            dx, dy = dx / length, dy / length
            # perpendicular vector
            px, py = -dy, dx
            gates.append(((cx - px*ROAD_RADIUS, cy - py*ROAD_RADIUS),
                          (cx + px*ROAD_RADIUS, cy + py*ROAD_RADIUS)))
        return gates

    def reset(self, offset=0):
        base_x, base_y = self.centerline[0]
        tx, ty = self.centerline[5]
        dx, dy = tx - base_x, ty - base_y
        length = math.hypot(dx, dy)
        dx, dy = dx / length, dy / length

        self.x = base_x - dx * offset * 20
        self.y = base_y - dy * offset * 20
        self.angle = math.degrees(math.atan2(dy, dx))
        self.speed = 0
        self.done = False
        self.distance = 0
        self.lifetime = 0
        self.next_gate = 0
        self.steps_since_gate = 0
        self.steps_since_improvement = 0
        self.last_fitness = 0

        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.car_surface = self.base_car_surface.copy()
        pygame.draw.rect(self.car_surface, self.color, (0, 0, 18, 36), border_radius=6)

        self.prev_pos = (self.x, self.y)
        return self._get_obs()

    def step(self, action):
        steer, throttle, brake = action
        steer = np.clip(steer, -1, 1)
        throttle = np.clip(throttle, 0, 1)
        brake = np.clip(brake, 0, 1)

        self.angle += steer * 4
        self.speed += throttle * 0.25
        self.speed -= brake * 0.3
        self.speed *= 0.98
        self.speed = np.clip(self.speed, 0, MAX_SPEED)

        old_pos = (self.x, self.y)
        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y += math.sin(math.radians(self.angle)) * self.speed

        reward = 0
        if not self._on_track(int(self.x), int(self.y)):
            self.done = True
            reward -= 200  # harsher penalty for crash
        else:
            # Forward alignment reward
            idx = self._nearest_center_idx()
            tx, ty = self.centerline[(idx+5) % len(self.centerline)]
            cx, cy = self.centerline[idx]
            tangent = (tx - cx, ty - cy)
            vel = (math.cos(math.radians(self.angle)) * self.speed,
                   math.sin(math.radians(self.angle)) * self.speed)
            dot = (vel[0]*tangent[0] + vel[1]*tangent[1]) / (np.linalg.norm(tangent)+1e-6)
            reward += max(0, dot * 0.05)

            # Small reward for moving forward
            reward += self.speed * 0.05

            # Survival bonus
            reward += 0.02

            # Anti-spin penalty (turning without forward motion)
            if self.speed < 1 and abs(steer) > 0.5:
                reward -= 0.5

            # Gate crossing
            p1, p2 = self.gates[self.next_gate]
            if self._crossed_line(old_pos, (self.x, self.y), p1, p2):
                reward += 200
                self.next_gate = (self.next_gate + 1) % len(self.gates)
                self.steps_since_gate = 0

        # Early termination if no improvement
        fitness_progress = self.distance + reward
        if fitness_progress > self.last_fitness:
            self.last_fitness = fitness_progress
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
        if self.steps_since_improvement > MAX_STEPS_WITHOUT_IMPROVEMENT:
            self.done = True

        self.distance += self.speed
        self.lifetime += 1
        self.prev_pos = (self.x, self.y)
        return self._get_obs(), reward, self.done, {}

    def _ccw(self, A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

    def _intersect(self, A, B, C, D):
        return self._ccw(A,C,D) != self._ccw(B,C,D) and self._ccw(A,B,C) != self._ccw(A,B,D)

    def _crossed_line(self, old_pos, new_pos, p1, p2):
        return self._intersect(old_pos, new_pos, p1, p2)

    def _nearest_center_idx(self):
        dists = [(i, (self.x - x)**2 + (self.y - y)**2) for i, (x, y) in enumerate(self.centerline)]
        return min(dists, key=lambda x: x[1])[0]

    def _on_track(self, x, y):
        if x < 0 or y < 0 or x >= SCREEN_SIZE[0] or y >= SCREEN_SIZE[1]:
            return False
        return self.track_mask.get_at((x, y))

    def _get_obs(self):
        sensors = []
        for d in [-75, -50, -25, 0, 25, 50, 75]:  # more sensors
            sensors.append(self._cast_ray(self.angle + d))
        sensors = np.array(sensors) / 150.0
        return np.append(sensors, self.speed / MAX_SPEED)

    def _cast_ray(self, ang):
        dx, dy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        for i in range(1, 150):
            px = int(self.x + dx * i * 4)
            py = int(self.y + dy * i * 4)
            if not self._on_track(px, py):
                return i * 4
        return 600

    def draw(self, screen):
        rotated = pygame.transform.rotate(self.car_surface, -self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        screen.blit(rotated, rect.topleft)

        # draw gates
        for (p1, p2) in self.gates:
            pygame.draw.line(screen, (255, 255, 0), p1, p2, 3)
