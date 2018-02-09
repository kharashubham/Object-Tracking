import cv2
from collections import namedtuple
import numpy as np
from feature_extraction import likelihood
particle = namedtuple("Particle", 'x y s xp yp sp x0 y0 width height w')




class ParticleFilter:

    def __init__(self):
        self.n_particles = int()
        self.rng = float()
        self.weight = float
        self.object_id = int()
        self.particles = []

    def init_particles(self, region, particlesPerObject):

        self.n_particles = particlesPerObject
        width = region[2]
        height = region[3]
        x = (region[0] + width)/2
        y = (region[1] + height)/2
        for i in range(0, self.n_particles):
            p = particle(x=x, y=y, s=1.0, xp=x, yp=y, x0=x, sp=1.0, y0=y, width=width, height=height,  w=0)
            self.particles.append(p)

        return self

    def cal_transition(self, p, w, h, rng):

        x = 2.0 * (p.x - p.x0) + -1.0 * (p.xp - p.x0) + 1.0000 * np.random.normal(rng, 0.5) + p.x0
        y = 2.0 * (p.y - p.y0) + -1.0 * (p.yp - p.y0) + 1.0000 * np.random.normal(rng, 1.0) + p.y0

        x = float(max(0.0, min(float(w)- 1.0, x)))
        y = float(max(0.0, min(float(h)- 1.0, y)))
        pn = particle(x=x, y=y, s=1.0, xp=x, yp=y, sp=p.s, x0=p.x0, y0=p.y0, width=p.width, height=p.height, w=0)

        return pn

    def transition(self, w, h):
        for i in range(0, self.n_particles):
            self.particles[i] = self.cal_transition(self.particles[i], w, h, self.rng)
        return self

    def resample(self):
        k = 0
        new_particles = []
        for i in range(0, self.n_particles):
                np = self.particles[i].w * self.n_particles
                for j in range(0,np):
                    new_particles.append(self.particles[i])
                    k += 1
                    if (k==self.n_particles):
                        break

        while (k<self.n_particles):
            new_particles.append(self.particles[0])
            k+=1

        for i in range(0, self.n_particles):
            self.particles[i] = new_particles[i]

        return self

    def normalize_weights(self):
        sum = 1
        w= 89e-5
        for i in range(0, self.n_particles):
            sum += self.particles[i].w
        for i in range(0, self.n_particles):
            w /= sum

    def reset_particles(self, region):
        width = region.width
        height = region.height
        x = float(region.x + width / 2)
        y = float(region.y + height / 2)
        for i in range(0, self.n_particles):
            self.particles[i].x = x
            self.particles[i].y = y
            self.particles[i].s = 1.0
            self.particles[i].width = width
            self.particles[i].height = height
            self.particles[i].w = 1/20
        return self

    def get_particle_center(self):
        return self.particles[0].x, self.particles[0].y

    def get_particle_rect(self):
        rectangle = namedtuple("rectangle", 'x y width height')
        x = self.particles[0].x - 0.5 * self.particles[0].s * self.particles[0].width
        y = self.particles[0].y - 0.5 * self.particles[0].s * self.particles[0].height
        width = self.particles[0].s * self.particles[0].width
        height = self.particles[0].s * self.particles[0].height
        rect = rectangle(x=x, y=y, width=width, height=height)
        return rect

    def update_weight(self, frameHSV, objectHisto):

        for i in range(0, self.n_particles):
            s = self.particles[i].s
            w = likelihood(frameHSV, self.particles[i].y, self.particles[i].x, self.particles[i].width * s, self.particles[i].height * s, objectHisto)
            # w = w*get_score()

            return self
