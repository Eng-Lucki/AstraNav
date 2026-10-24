"""
Phase 4: Inertial navigation (dead reckoning) with star-tracker aiding.

The vehicle flies at constant speed while slowly turning. Its gyro has a
small constant bias plus noise, so integrating it makes the heading
estimate drift, and a wrong heading makes the position error grow.

Two navigators run side by side on the same gyro data:
  * INS only   - pure dead reckoning, drifts without limit
  * INS + star - every star fix corrects the heading and refines an
                 estimate of the gyro bias, so drift stays small

Simplification: the star camera is fixed to the vehicle, so a change in
vehicle heading shows up as a roll of the star field in the image. The
solver's measured roll therefore gives the heading directly.
"""
import math
import numpy as np

METERS_PER_DEG_LAT = 111_320.0


def wrap_angle(a):
    return (a + 180.0) % 360.0 - 180.0


class InertialNavigator:
    def __init__(self, lat0=33.3152, lon0=44.3661, heading0=42.5, speed=250.0,
                 gyro_bias=0.02, gyro_noise=0.003, seed=None):
        self.lat0, self.lon0 = lat0, lon0
        self.heading0 = heading0
        self.speed = speed               # m/s
        self.gyro_bias = gyro_bias       # deg/s, unknown to the navigator
        self.gyro_noise = gyro_noise     # deg/s per sample
        self.heading_gain = 0.9          # how much a star fix is trusted
        self.bias_gain = 0.05
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.t = 0.0
        self.true_heading = self.heading0
        self.ins_heading = self.heading0
        self.aided_heading = self.heading0
        self.bias_estimate = 0.0
        self.last_fix_t = None
        self.true_pos = np.zeros(2)      # north, east (m)
        self.ins_pos = np.zeros(2)
        self.aided_pos = np.zeros(2)
        self.history = {k: [] for k in ("t", "ins_heading_err", "aided_heading_err",
                                        "ins_pos_err", "aided_pos_err", "fix")}

    def true_turn_rate(self, t):
        """Gentle S-turns: about +/-15 deg of heading change every 3 minutes."""
        return 0.5 * math.sin(2 * math.pi * t / 180.0)

    def predict(self, dt):
        """Advance true motion and both dead-reckoning solutions by dt seconds."""
        rate = self.true_turn_rate(self.t)
        gyro = rate + self.gyro_bias + self.rng.normal(0, self.gyro_noise)

        self.true_heading = wrap_angle(self.true_heading + rate * dt)
        self.ins_heading = wrap_angle(self.ins_heading + gyro * dt)
        self.aided_heading = wrap_angle(self.aided_heading + (gyro - self.bias_estimate) * dt)

        for pos, hdg in ((self.true_pos, self.true_heading),
                         (self.ins_pos, self.ins_heading),
                         (self.aided_pos, self.aided_heading)):
            h = math.radians(hdg)
            pos += self.speed * dt * np.array([math.cos(h), math.sin(h)])
        self.t += dt

    def star_fix(self, measured_heading):
        """Correct the aided navigator with a heading measured from the stars."""
        innovation = wrap_angle(measured_heading - self.aided_heading)
        if self.last_fix_t is not None and self.t > self.last_fix_t:
            # Heading drifted by (bias - bias_estimate) * dt since the last fix,
            # so the innovation tells us how wrong the bias estimate is.
            self.bias_estimate -= self.bias_gain * innovation / (self.t - self.last_fix_t)
        self.aided_heading = wrap_angle(self.aided_heading + self.heading_gain * innovation)
        self.last_fix_t = self.t

    def record(self, fix):
        h = self.history
        h["t"].append(self.t)
        h["ins_heading_err"].append(wrap_angle(self.ins_heading - self.true_heading))
        h["aided_heading_err"].append(wrap_angle(self.aided_heading - self.true_heading))
        h["ins_pos_err"].append(float(np.linalg.norm(self.ins_pos - self.true_pos)))
        h["aided_pos_err"].append(float(np.linalg.norm(self.aided_pos - self.true_pos)))
        h["fix"].append(bool(fix))

    def to_latlon(self, pos):
        lat = self.lat0 + pos[0] / METERS_PER_DEG_LAT
        lon = self.lon0 + pos[1] / (METERS_PER_DEG_LAT * math.cos(math.radians(self.lat0)))
        return lat, lon
