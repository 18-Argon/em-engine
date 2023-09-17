import tkinter as tk
import multiprocessing
import typing
import numpy as np
import math
import pyglet as pyg

objects = []
bg_batch = pyg.graphics.Batch()
grid_batch = pyg.graphics.Batch()
fg_batch = pyg.graphics.Batch()

DT = 1E-6
TIME_SCALE = 10
EPS0 = 8.85418E-6
U0 = 1.25663
DRAG_COEF = 0
BG_E = np.array([0, 0])

K = 1 / (4 * math.pi * EPS0)
C = 1 / math.sqrt(EPS0 * U0)


class Object:
    mass: float
    charge: float
    pos = np.array([0, 0])
    vel = np.array([0, 0])
    acc = np.array([0, 0])

    is_kinematic: bool

    def __init__(self, pos=np.array([0, 0]), mass=1, charge=1, vel=np.array([0, 0]), acc=np.array([0, 0]),
                 is_kinematic=False):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.mass = mass
        self.is_kinematic = is_kinematic
        self.charge = charge
        objects.append(self)

    def is_colliding(self):
        return False

    def elastic_collide(self, other):
        return

    def get_forces(self,dt):
        if self.is_kinematic:
            return np.array([0, 0])

        electric_force = np.array([0., 0.])
        drag_force = -DRAG_COEF * self.vel

        for object in objects:
            if object == self:
                continue
            r = self.pos - object.pos
            r_hat = r / np.sqrt(np.dot(r, r))


                # electric_force -= r_hat * self.charge * object.charge * K / (np.dot(r, r))
            # else:
            electric_force += r_hat * self.charge * object.charge * K / (np.dot(r, r))

        F_net = (electric_force + drag_force + self.charge * BG_E)

        if self.is_colliding(object, r):
            self.elastic_collide(object, r_hat, dt)
        return F_net

    def update_pos(self, dt):
        dt *= TIME_SCALE
        new_pos = self.pos + self.vel * dt  # + self.acc * dt * dt * 0.5
        new_acc = self.get_forces(dt) / self.mass
        new_vel = self.vel + (self.acc + new_acc) * (dt * 0.5)

        self.pos = new_pos
        self.vel = new_vel
        self.acc = new_acc


class CircObject(Object):
    radius: float
    do_collision: bool
    gfx: pyg.shapes.Circle

    def __init__(self, radius=10, color=(200, 50, 50), pos=np.array([0, 0]), mass=1, charge=1, vel=np.array([0, 0]),
                 acc=np.array([0, 0]), is_kinematic=False):
        super().__init__(pos=pos, mass=mass, charge=charge, vel=vel, acc=acc, is_kinematic=is_kinematic)
        self.radius = radius
        self.gfx = pyg.shapes.Circle(self.pos[0], self.pos[1], self.radius, color=color, batch=fg_batch)

    def is_colliding(self, other, r: np.array):
        is_coll = np.dot(r, r) <= math.pow((self.radius + other.radius), 2)
        print(is_coll)
        return is_coll

    def elastic_collide(self, other, r_hat, dt):
        # Simultaneity is required
        new_vel = ((self.mass - other.mass) * self.vel + 2 * other.mass * other.vel) / (self.mass + other.mass)
        other.vel = ((other.mass - self.mass) * other.vel + 2 * self.mass * self.vel) / (self.mass + other.mass)
        # other.update_pos(dt)
        # self.vel=new_vel
        # Add impulses into every other object?
        pass

def run():
    siml_window = pyg.window.Window(800, 600)

    # Draw grid
    grid_lines = []

    for x in range(17):
        rect_x = pyg.shapes.Rectangle(x * 50 - 1, 0, 2, 600, batch=grid_batch, color=(30, 30, 30))
        grid_lines.append(rect_x)
        for y in range(13):
            rect_y = pyg.shapes.Rectangle(0, y * 50 - 1, 800, 2, batch=grid_batch, color=(30, 30, 30))
            grid_lines.append(rect_y)

    c1 = CircObject(pos=np.array([400, 400]), charge=0, is_kinematic=False)
    c2 = CircObject(pos=np.array([200, 200]), mass=1,vel=np.array([10, 10]), color=(0, 255, 0))

    # @siml_window.event
    def on_draw():
        siml_window.clear()
        bg_batch.draw()
        grid_batch.draw()
        fg_batch.draw()

    def update(dt):
        on_draw()
        for object in objects:
            object.update_pos(dt)
            object.gfx.x, object.gfx.y = object.pos[0], object.pos[1]
            print(object.gfx.color,object.vel)

    pyg.clock.schedule_interval(update, DT)
    pyg.app.run()


if __name__ == "__main__":
    run()
