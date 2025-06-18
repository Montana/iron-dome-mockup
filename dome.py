import math
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
from datetime import datetime, timedelta
import json

class TargetType(Enum):
    ROCKET = "rocket"
    MORTAR = "mortar"
    ARTILLERY = "artillery"
    CRUISE_MISSILE = "cruise_missile"
    BALLISTIC_MISSILE = "ballistic_missile"
    DRONE = "drone"
    AIRCRAFT = "aircraft"
    DECOY = "decoy"
    CLUSTER_MUNITION = "cluster_munition"

class RadarType(Enum):
    SEARCH = "search"
    TRACKING = "tracking"
    FIRE_CONTROL = "fire_control"
    EARLY_WARNING = "early_warning"
    PHASED_ARRAY = "phased_array"

class InterceptorType(Enum):
    TAMIR = "tamir"
    DAVID_SLING = "david_sling"
    ARROW_2 = "arrow_2"
    ARROW_3 = "arrow_3"
    IRON_BEAM = "iron_beam"
    PATRIOT = "patriot"

class WeatherCondition(Enum):
    CLEAR = "clear"
    LIGHT_CLOUDS = "light_clouds"
    HEAVY_CLOUDS = "heavy_clouds"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    SANDSTORM = "sandstorm"
    FOG = "fog"
    SNOW = "snow"

class ThreatPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    IMMINENT = 5

class SystemMode(Enum):
    NORMAL = "normal"
    HIGH_ALERT = "high_alert"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"
    TEST = "test"
    COMBAT = "combat"

class TrackQuality(Enum):
    INITIAL = 0.2
    TENTATIVE = 0.4
    CONFIRMED = 0.6
    HIGH = 0.8
    LOCK = 0.95
    PRECISION = 0.99

class ECMType(Enum):
    JAMMING = "jamming"
    CHAFF = "chaff"
    FLARE = "flare"
    DECOY = "decoy"
    STEALTH = "stealth"

GRAVITY = 9.81
AIR_RESISTANCE_COEFF = 0.47
AIR_DENSITY = 1.225
SPEED_OF_SOUND = 343
EARTH_RADIUS = 6371000
CURVATURE_FACTOR = 1.33

RADAR_FREQUENCY_BANDS = {
    RadarType.SEARCH: (1.2, 1.4),
    RadarType.TRACKING: (2.9, 3.1),
    RadarType.FIRE_CONTROL: (8.5, 10.5),
    RadarType.EARLY_WARNING: (0.4, 0.45),
    RadarType.PHASED_ARRAY: (2.9, 3.1)
}

TARGET_CHARACTERISTICS = {
    TargetType.ROCKET: {
        "rcs_base": 0.08,
        "rcs_variation": 0.04,
        "mass_range": (15, 45),
        "diameter": 0.12,
        "length": 3.2,
        "max_speed": 850,
        "burnout_time": 8.5,
        "stages": 1,
        "maneuverability": 0.1,
        "ecm_capability": 0.0,
        "typical_range": 40000
    },
    TargetType.MORTAR: {
        "rcs_base": 0.03,
        "rcs_variation": 0.02,
        "mass_range": (3, 12),
        "diameter": 0.08,
        "length": 0.6,
        "max_speed": 320,
        "burnout_time": 0.5,
        "stages": 1,
        "maneuverability": 0.0,
        "ecm_capability": 0.0,
        "typical_range": 8000
    },
    TargetType.BALLISTIC_MISSILE: {
        "rcs_base": 0.5,
        "rcs_variation": 0.3,
        "mass_range": (800, 2000),
        "diameter": 0.9,
        "length": 12.0,
        "max_speed": 2500,
        "burnout_time": 60,
        "stages": 2,
        "maneuverability": 0.3,
        "ecm_capability": 0.7,
        "typical_range": 300000
    },
    TargetType.CRUISE_MISSILE: {
        "rcs_base": 0.02,
        "rcs_variation": 0.01,
        "mass_range": (200, 600),
        "diameter": 0.35,
        "length": 5.5,
        "max_speed": 280,
        "burnout_time": 3600,
        "stages": 1,
        "maneuverability": 0.8,
        "ecm_capability": 0.9,
        "typical_range": 2500000
    },
    TargetType.DRONE: {
        "rcs_base": 0.01,
        "rcs_variation": 0.008,
        "mass_range": (2, 25),
        "diameter": 0.15,
        "length": 1.8,
        "max_speed": 180,
        "burnout_time": 7200,
        "stages": 1,
        "maneuverability": 0.9,
        "ecm_capability": 0.4,
        "typical_range": 150000
    },
    TargetType.ARTILLERY: {
        "rcs_base": 0.06,
        "rcs_variation": 0.03,
        "mass_range": (20, 100),
        "diameter": 0.155,
        "length": 0.8,
        "max_speed": 900,
        "burnout_time": 2.0,
        "stages": 1,
        "maneuverability": 0.05,
        "ecm_capability": 0.0,
        "typical_range": 30000
    },
    TargetType.AIRCRAFT: {
        "rcs_base": 15.0,
        "rcs_variation": 8.0,
        "mass_range": (8000, 80000),
        "diameter": 4.0,
        "length": 20.0,
        "max_speed": 600,
        "burnout_time": 18000,
        "stages": 1,
        "maneuverability": 0.95,
        "ecm_capability": 0.8,
        "typical_range": 5000000
    },
    TargetType.DECOY: {
        "rcs_base": 0.15,
        "rcs_variation": 0.1,
        "mass_range": (5, 15),
        "diameter": 0.2,
        "length": 1.5,
        "max_speed": 400,
        "burnout_time": 30,
        "stages": 1,
        "maneuverability": 0.6,
        "ecm_capability": 0.9,
        "typical_range": 20000
    },
    TargetType.CLUSTER_MUNITION: {
        "rcs_base": 0.04,
        "rcs_variation": 0.02,
        "mass_range": (40, 200),
        "diameter": 0.2,
        "length": 2.0,
        "max_speed": 750,
        "burnout_time": 5.0,
        "stages": 1,
        "maneuverability": 0.0,
        "ecm_capability": 0.0,
        "typical_range": 50000
    }
}

WEATHER_RADAR_EFFECTS = {
    WeatherCondition.CLEAR: {"attenuation": 0.0, "noise": 0.0, "beam_bending": 0.0},
    WeatherCondition.LIGHT_CLOUDS: {"attenuation": 0.1, "noise": 0.05, "beam_bending": 0.02},
    WeatherCondition.HEAVY_CLOUDS: {"attenuation": 0.25, "noise": 0.15, "beam_bending": 0.05},
    WeatherCondition.LIGHT_RAIN: {"attenuation": 0.3, "noise": 0.2, "beam_bending": 0.08},
    WeatherCondition.HEAVY_RAIN: {"attenuation": 0.6, "noise": 0.4, "beam_bending": 0.15},
    WeatherCondition.THUNDERSTORM: {"attenuation": 0.8, "noise": 0.7, "beam_bending": 0.25},
    WeatherCondition.SANDSTORM: {"attenuation": 0.7, "noise": 0.5, "beam_bending": 0.3},
    WeatherCondition.FOG: {"attenuation": 0.4, "noise": 0.3, "beam_bending": 0.1},
    WeatherCondition.SNOW: {"attenuation": 0.35, "noise": 0.25, "beam_bending": 0.12}
}

INTERCEPTOR_CAPABILITIES = {
    InterceptorType.TAMIR: {
        "max_range": 10000,
        "max_altitude": 10000,
        "min_range": 500,
        "max_speed": 1500,
        "acceleration": 150,
        "maneuverability": 40,
        "success_rate": 0.88,
        "reload_time": 12,
        "cost": 50000,
        "warhead_type": "fragmentation",
        "guidance": "active_radar",
        "datalink": True,
        "countermeasure_resistance": 0.7
    },
    InterceptorType.DAVID_SLING: {
        "max_range": 70000,
        "max_altitude": 40000,
        "min_range": 5000,
        "max_speed": 2200,
        "acceleration": 200,
        "maneuverability": 25,
        "success_rate": 0.92,
        "reload_time": 25,
        "cost": 1000000,
        "warhead_type": "fragmentation",
        "guidance": "active_radar",
        "datalink": True,
        "countermeasure_resistance": 0.85
    },
    InterceptorType.ARROW_2: {
        "max_range": 90000,
        "max_altitude": 50000,
        "min_range": 15000,
        "max_speed": 2500,
        "acceleration": 180,
        "maneuverability": 15,
        "success_rate": 0.90,
        "reload_time": 45,
        "cost": 3000000,
        "warhead_type": "fragmentation",
        "guidance": "command",
        "datalink": True,
        "countermeasure_resistance": 0.8
    },
    InterceptorType.ARROW_3: {
        "max_range": 150000,
        "max_altitude": 100000,
        "min_range": 30000,
        "max_speed": 3500,
        "acceleration": 220,
        "maneuverability": 10,
        "success_rate": 0.95,
        "reload_time": 60,
        "cost": 4500000,
        "warhead_type": "kinetic",
        "guidance": "inertial_terminal",
        "datalink": True,
        "countermeasure_resistance": 0.9
    },
    InterceptorType.IRON_BEAM: {
        "max_range": 15000,
        "max_altitude": 8000,
        "min_range": 100,
        "max_speed": 299792458,
        "acceleration": 0,
        "maneuverability": 0,
        "success_rate": 0.95,
        "reload_time": 3,
        "cost": 5000,
        "warhead_type": "laser",
        "guidance": "optical",
        "datalink": False,
        "countermeasure_resistance": 0.95
    },
    InterceptorType.PATRIOT: {
        "max_range": 70000,
        "max_altitude": 25000,
        "min_range": 3000,
        "max_speed": 1800,
        "acceleration": 160,
        "maneuverability": 20,
        "success_rate": 0.85,
        "reload_time": 30,
        "cost": 4000000,
        "warhead_type": "fragmentation",
        "guidance": "command",
        "datalink": True,
        "countermeasure_resistance": 0.75
    }
}

@dataclass
class TerrainModel:
    elevation_map: Dict[Tuple[int, int], float] = field(default_factory=dict)
    clutter_map: Dict[Tuple[int, int], float] = field(default_factory=dict)
    urban_areas: List[Tuple[float, float, float]] = field(default_factory=list)
    
    def __post_init__(self):
        self.generate_terrain()
    
    def generate_terrain(self):
        for x in range(-30, 31):
            for y in range(-30, 31):
                elevation = random.uniform(0, 300) + 50 * math.sin(x * 0.1) * math.cos(y * 0.1)
                self.elevation_map[(x, y)] = max(0, elevation)
                
                if random.random() < 0.3:
                    self.clutter_map[(x, y)] = random.uniform(0.001, 0.01)
                else:
                    self.clutter_map[(x, y)] = random.uniform(0.0001, 0.001)
        
        self.urban_areas = [(5000, 0, 2000), (8000, 2000, 1500), (3000, -1000, 1800)]
    
    def get_elevation(self, x: float, y: float) -> float:
        grid_x, grid_y = int(x // 1000), int(y // 1000)
        return self.elevation_map.get((grid_x, grid_y), 0.0)
    
    def get_clutter_rcs(self, x: float, y: float) -> float:
        grid_x, grid_y = int(x // 1000), int(y // 1000)
        return self.clutter_map.get((grid_x, grid_y), 0.001)
    
    def calculate_line_of_sight(self, radar_pos: Tuple[float, float, float], 
                               target_pos: Tuple[float, float, float]) -> bool:
        rx, ry, rz = radar_pos
        tx, ty, tz = target_pos
        
        distance = math.sqrt((tx - rx)**2 + (ty - ry)**2)
        if distance > 100000:
            return False
        
        earth_curvature = distance**2 / (2 * EARTH_RADIUS * CURVATURE_FACTOR)
        required_height = rz + (tz - rz) * 0.5 + earth_curvature
        
        mid_x, mid_y = (rx + tx) / 2, (ry + ty) / 2
        terrain_height = self.get_elevation(mid_x, mid_y)
        
        return required_height > terrain_height + 50

@dataclass
class AtmosphericModel:
    temperature: float = 15.0
    pressure: float = 1013.25
    humidity: float = 60.0
    wind_speed: float = 5.0
    wind_direction: float = 270.0
    turbulence_level: float = 0.1
    
    def get_air_density(self, altitude: float) -> float:
        temp_k = self.temperature + 273.15 - 0.0065 * altitude
        pressure_alt = self.pressure * (1 - 0.0065 * altitude / temp_k) ** 5.255
        return pressure_alt * 100 / (287.05 * temp_k)
    
    def get_sound_speed(self, altitude: float) -> float:
        temp_k = self.temperature + 273.15 - 0.0065 * altitude
        return math.sqrt(1.4 * 287.05 * temp_k)
    
    def apply_wind_effect(self, vx: float, vy: float, vz: float, altitude: float) -> Tuple[float, float, float]:
        wind_factor = min(1.0, altitude / 10000)
        wind_vx = self.wind_speed * math.cos(math.radians(self.wind_direction)) * wind_factor
        wind_vy = self.wind_speed * math.sin(math.radians(self.wind_direction)) * wind_factor
        
        turbulence_vx = random.gauss(0, self.turbulence_level * wind_factor)
        turbulence_vy = random.gauss(0, self.turbulence_level * wind_factor)
        turbulence_vz = random.gauss(0, self.turbulence_level * wind_factor * 0.5)
        
        return (vx + wind_vx * 0.1 + turbulence_vx,
                vy + wind_vy * 0.1 + turbulence_vy,
                vz + turbulence_vz)

class AdvancedTarget:
    def __init__(self, target_type: TargetType = None):
        if target_type is None:
            target_type = random.choice(list(TargetType))
        
        self.type = target_type
        self.characteristics = TARGET_CHARACTERISTICS[target_type]
        self.track_id = random.randint(10000, 99999)
        
        self.x0 = random.uniform(-35000, -5000)
        self.y0 = random.uniform(-20000, 20000)
        self.z0 = random.uniform(100, 2000)
        self.x, self.y, self.z = self.x0, self.y0, self.z0
        
        self.mass = random.uniform(*self.characteristics["mass_range"])
        self.initial_mass = self.mass
        self.fuel_mass = self.mass * 0.6
        self.dry_mass = self.mass * 0.4
        self.burn_rate = self.fuel_mass / self.characteristics["burnout_time"] if self.characteristics["burnout_time"] > 0 else 0
        
        target_x = random.uniform(2000, 10000)
        target_y = random.uniform(-5000, 5000)
        
        range_to_target = math.sqrt((target_x - self.x0)**2 + (target_y - self.y0)**2)
        optimal_angle = math.atan2(math.sqrt(range_to_target * GRAVITY), range_to_target)
        launch_angle = random.gauss(optimal_angle, math.radians(10))
        launch_azimuth = math.atan2(target_y - self.y0, target_x - self.x0)
        
        launch_speed = min(self.characteristics["max_speed"], 
                          math.sqrt(range_to_target * GRAVITY / math.sin(2 * launch_angle)))
        
        self.vx = launch_speed * math.cos(launch_angle) * math.cos(launch_azimuth)
        self.vy = launch_speed * math.cos(launch_angle) * math.sin(launch_azimuth)
        self.vz = launch_speed * math.sin(launch_angle)
        
        self.current_stage = 1
        self.stages_remaining = self.characteristics["stages"]
        self.stage_separation_times = []
        for i in range(1, self.characteristics["stages"] + 1):
            sep_time = self.characteristics["burnout_time"] * (i / self.characteristics["stages"])
            self.stage_separation_times.append(sep_time)
        
        self.base_rcs = self.characteristics["rcs_base"]
        self.current_rcs = self.base_rcs
        self.rcs_enhancement_factor = random.uniform(0.8, 1.2)
        
        self.t = 0.0
        self.burning = True
        self.separated_stages = []
        
        self.detected = False
        self.track_quality = TrackQuality.INITIAL.value
        self.detection_time = None
        self.last_radar_update = 0.0
        self.track_covariance = np.eye(6) * 100
        self.position_history = []
        self.velocity_history = []
        
        self.ecm_active = False
        self.ecm_types = []
        if random.random() < self.characteristics["ecm_capability"]:
            self.ecm_active = True
            available_ecm = [ECMType.JAMMING, ECMType.CHAFF, ECMType.DECOY]
            self.ecm_types = random.sample(available_ecm, random.randint(1, len(available_ecm)))
        
        self.maneuvering = False
        self.maneuver_time_remaining = 0.0
        self.maneuver_acceleration = (0.0, 0.0, 0.0)
        
        self.predicted_impact_point = None
        self.predicted_impact_time = None
        self.threat_priority = ThreatPriority.LOW
        
        self.fragments = []
        self.is_fragment = False
        self.parent_track_id = None
        
        self.signature_temperature = 300 + random.uniform(-50, 200)
        self.optical_signature = random.uniform(0.1, 1.0)
        
        self.classification_confidence = 0.0
        self.false_alarm_probability = 0.05
        
        self.intercept_attempts = 0
        self.max_intercept_attempts = 3
        
    def update_physics(self, dt: float, atmosphere: AtmosphericModel):
        old_pos = (self.x, self.y, self.z)
        
        if self.burning and self.fuel_mass > 0:
            fuel_consumed = min(self.burn_rate * dt, self.fuel_mass)
            self.fuel_mass -= fuel_consumed
            self.mass = self.dry_mass + self.fuel_mass
            
            if self.fuel_mass <= 0:
                self.burning = False
        
        air_density = atmosphere.get_air_density(self.z)
        speed = math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        
        if speed > 0:
            drag_force = 0.5 * air_density * speed**2 * AIR_RESISTANCE_COEFF * math.pi * (self.characteristics["diameter"]/2)**2
            drag_acceleration = drag_force / self.mass
            
            drag_ax = -drag_acceleration * (self.vx / speed)
            drag_ay = -drag_acceleration * (self.vy / speed)
            drag_az = -drag_acceleration * (self.vz / speed)
        else:
            drag_ax = drag_ay = drag_az = 0
        
        if self.maneuvering and self.maneuver_time_remaining > 0:
            self.maneuver_time_remaining -= dt
            maneuver_ax, maneuver_ay, maneuver_az = self.maneuver_acceleration
        else:
            self.maneuvering = False
            maneuver_ax = maneuver_ay = maneuver_az = 0
        
        total_ax = drag_ax + maneuver_ax
        total_ay = drag_ay + maneuver_ay
        total_az = -GRAVITY + drag_az + maneuver_az
        
        self.vx, self.vy, self.vz = atmosphere.apply_wind_effect(
            self.vx + total_ax * dt,
            self.vy + total_ay * dt,
            self.vz + total_az * dt,
            self.z
        )
        
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        
        self.t += dt
        
        if len(self.stage_separation_times) > 0 and self.t >= self.stage_separation_times[0]:
            self.separate_stage()
        
        if random.random() < self.characteristics["maneuverability"] * dt:
            self.initiate_maneuver()
        
        self.position_history.append((self.x, self.y, self.z, self.t))
        self.velocity_history.append((self.vx, self.vy, self.vz, self.t))
        
        if len(self.position_history) > 1000:
            self.position_history.pop(0)
            self.velocity_history.pop(0)
        
        self.update_rcs()
        self.update_prediction()
        self.update_threat_assessment()
        self.check_fragmentation()
    
    def separate_stage(self):
        if self.stages_remaining > 1:
            stage_mass = self.mass * 0.3
            self.mass -= stage_mass
            self.stages_remaining -= 1
            self.current_stage += 1
            self.stage_separation_times.pop(0)
            
            debris_velocity = (
                self.vx + random.uniform(-50, 50),
                self.vy + random.uniform(-50, 50),
                self.vz + random.uniform(-100, -50)
            )
            
            debris = {
                'position': (self.x, self.y, self.z),
                'velocity': debris_velocity,
                'mass': stage_mass,
                'rcs': 0.05,
                'time': self.t
            }
            self.separated_stages.append(debris)
    
    def initiate_maneuver(self):
        if self.characteristics["maneuverability"] > 0.5:
            self.maneuvering = True
            self.maneuver_time_remaining = random.uniform(0.5, 3.0)
            
            max_g = self.characteristics["maneuverability"] * 9.81
            maneuver_ax = random.uniform(-max_g, max_g)
            maneuver_ay = random.uniform(-max_g, max_g)
            maneuver_az = random.uniform(-max_g/2, max_g/2)
            
            self.maneuver_acceleration = (maneuver_ax, maneuver_ay, maneuver_az)
    
    def update_rcs(self):
        base_rcs = self.base_rcs * self.rcs_enhancement_factor
        
        speed = math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        aspect_angle = math.atan2(math.sqrt(self.vx**2 + self.vy**2), abs(self.vz))
        aspect_factor = 1.0 + 0.5 * math.sin(aspect_angle)
        
        altitude_factor = 1.0 + (self.z / 20000) * 0.2
        
        if self.burning:
            thermal_factor = 1.5
        else:
            thermal_factor = 1.0
        
        ecm_factor = 1.0
        if self.ecm_active:
            for ecm_type in self.ecm_types:
                if ecm_type == ECMType.CHAFF:
                    ecm_factor *= random.uniform(2.0, 5.0)
                elif ecm_type == ECMType.STEALTH:
                    ecm_factor *= random.uniform(0.1, 0.3)
        
        self.current_rcs = base_rcs * aspect_factor * altitude_factor * thermal_factor * ecm_factor
        
        if self.current_rcs < 0.001:
            self.current_rcs = 0.001
    
    def update_prediction(self):
        if len(self.position_history) < 3:
            return
        
        recent_positions = self.position_history[-5:]
        if len(recent_positions) < 2:
            return
        
        dt_avg = (recent_positions[-1][3] - recent_positions[0][3]) / (len(recent_positions) - 1)
        if dt_avg <= 0:
            return
        
        vx_avg = (recent_positions[-1][0] - recent_positions[0][0]) / (recent_positions[-1][3] - recent_positions[0][3])
        vy_avg = (recent_positions[-1][1] - recent_positions[0][1]) / (recent_positions[-1][3] - recent_positions[0][3])
        vz_avg = (recent_positions[-1][2] - recent_positions[0][2]) / (recent_positions[-1][3] - recent_positions[0][3])
        
        if vz_avg >= 0:
            self.predicted_impact_time = float('inf')
            return
        
        time_to_ground = -self.z / vz_avg
        if time_to_ground < 0:
            time_to_ground = (-vz_avg + math.sqrt(vz_avg**2 + 2 * GRAVITY * self.z)) / GRAVITY
        
        impact_x = self.x + vx_avg * time_to_ground
        impact_y = self.y + vy_avg * time_to_ground
        
        self.predicted_impact_point = (impact_x, impact_y)
        self.predicted_impact_time = time_to_ground
    
    def update_threat_assessment(self):
        if self.predicted_impact_point is None:
            self.threat_priority = ThreatPriority.LOW
            return
        
        impact_x, impact_y = self.predicted_impact_point
        
        critical_zones = [(5000, 0, 2000), (8000, 2000, 1500), (3000, -1000, 1800)]
        min_distance = float('inf')
        
        for zone_x, zone_y, zone_radius in critical_zones:
            distance = math.sqrt((impact_x - zone_x)**2 + (impact_y - zone_y)**2)
            min_distance = min(min_distance, distance)
        
        if min_distance < 500:
            self.threat_priority = ThreatPriority.IMMINENT
        elif min_distance < 1000:
            self.threat_priority = ThreatPriority.CRITICAL
        elif min_distance < 2500:
            self.threat_priority = ThreatPriority.HIGH
        elif min_distance < 5000:
            self.threat_priority = ThreatPriority.MEDIUM
        else:
            self.threat_priority = ThreatPriority.LOW
        
        if self.type == TargetType.BALLISTIC_MISSILE:
            if self.threat_priority.value < 4:
                self.threat_priority = ThreatPriority(min(5, self.threat_priority.value + 1))
        elif self.type == TargetType.DECOY:
            if self.threat_priority.value > 2:
                self.threat_priority = ThreatPriority(max(1, self.threat_priority.value - 1))
    
    def check_fragmentation(self):
        if self.type == TargetType.CLUSTER_MUNITION and self.z < 3000 and not self.fragments:
            num_fragments = random.randint(8, 24)
            for _ in range(num_fragments):
                fragment = AdvancedTarget(TargetType.ARTILLERY)
                fragment.x = self.x + random.uniform(-100, 100)
                fragment.y = self.y + random.uniform(-100, 100)
                fragment.z = self.z + random.uniform(-50, 50)
                fragment.vx = self.vx + random.uniform(-200, 200)
                fragment.vy = self.vy + random.uniform(-200, 200)
                fragment.vz = self.vz + random.uniform(-100, 50)
                fragment.is_fragment = True
                fragment.parent_track_id = self.track_id
                fragment.mass = self.mass / num_fragments
                self.fragments.append(fragment)
    
    def get_detection_probability(self, radar_type: RadarType, distance: float, weather: WeatherCondition) -> float:
        base_prob = 0.95
        
        rcs_factor = min(1.0, self.current_rcs / 0.1)
        range_factor = max(0.1, 1.0 - (distance / 50000))
        
        weather_effects = WEATHER_RADAR_EFFECTS[weather]
        weather_factor = 1.0 - weather_effects["attenuation"]
        
        if self.ecm_active and ECMType.JAMMING in self.ecm_types:
            ecm_factor = 0.3
        else:
            ecm_factor = 1.0
        
        stealth_factor = 1.0
        if self.ecm_active and ECMType.STEALTH in self.ecm_types:
            stealth_factor = 0.2
        
        return base_prob * rcs_factor * range_factor * weather_factor * ecm_factor * stealth_factor

class AdvancedRadar:
    def __init__(self, x: float, y: float, z: float, radar_type: RadarType):
        self.x = x
        self.y = y
        self.z = z
        self.radar_type = radar_type
        self.frequency_min, self.frequency_max = RADAR_FREQUENCY_BANDS[radar_type]
        self.current_frequency = random.uniform(self.frequency_min, self.frequency_max)
        
        if radar_type == RadarType.EARLY_WARNING:
            self.detection_range = 400000
            self.track_capacity = 500
            self.beam_width = 6.0
            self.update_rate = 12.0
        elif radar_type == RadarType.SEARCH:
            self.detection_range = 150000
            self.track_capacity = 200
            self.beam_width = 3.0
            self.update_rate = 4.0
        elif radar_type == RadarType.TRACKING:
            self.detection_range = 80000
            self.track_capacity = 100
            self.beam_width = 1.5
            self.update_rate = 1.0
        elif radar_type == RadarType.FIRE_CONTROL:
            self.detection_range = 40000
            self.track_capacity = 20
            self.beam_width = 0.5
            self.update_rate = 0.1
        elif radar_type == RadarType.PHASED_ARRAY:
            self.detection_range = 200000
            self.track_capacity = 1000
            self.beam_width = 2.0
            self.update_rate = 0.5
        
        self.power_output = random.uniform(0.8, 1.0)
        self.status = "OPERATIONAL"
        self.maintenance_time = 0.0
        self.tracks = {}
        self.last_update = 0.0
        self.scan_pattern = "circular"
        self.current_azimuth = 0.0
        self.elevation_angle = 15.0
        
        self.jamming_resistance = random.uniform(0.6, 0.9)
        self.frequency_agility = radar_type in [RadarType.PHASED_ARRAY, RadarType.FIRE_CONTROL]
        self.adaptive_processing = radar_type == RadarType.PHASED_ARRAY
        
        self.clutter_map = {}
        self.false_alarm_rate = 0.001
        self.detection_threshold = 0.7
        
        self.performance_degradation = 0.0
        self.calibration_drift = 0.0
        
    def can_detect(self, target: AdvancedTarget, weather: WeatherCondition, terrain: TerrainModel) -> Tuple[bool, float]:
        distance_3d = math.sqrt((target.x - self.x)**2 + (target.y - self.y)**2 + (target.z - self.z)**2)
        
        if distance_3d > self.detection_range:
            return False, 0.0
        
        if not terrain.calculate_line_of_sight((self.x, self.y, self.z), (target.x, target.y, target.z)):
            return False, 0.0
        
        if self.status != "OPERATIONAL":
            return False, 0.0
        
        detection_prob = target.get_detection_probability(self.radar_type, distance_3d, weather)
        
        performance_factor = (1.0 - self.performance_degradation) * self.power_output
        detection_prob *= performance_factor
        
        angle_to_target = math.atan2(target.y - self.y, target.x - self.x)
        beam_center = math.radians(self.current_azimuth)
        angle_diff = abs(angle_to_target - beam_center)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        if angle_diff > math.radians(self.beam_width):
            detection_prob *= 0.1
        
        clutter_rcs = terrain.get_clutter_rcs(target.x, target.y)
        if target.current_rcs < clutter_rcs * 3:
            detection_prob *= 0.3
        
        if random.random() < self.false_alarm_rate:
            return True, 0.3
        
        return random.random() < detection_prob, detection_prob
    
    def update_track(self, target: AdvancedTarget, detection_confidence: float):
        if target.track_id not in self.tracks:
            self.tracks[target.track_id] = {
                'first_detection': time.time(),
                'last_update': time.time(),
                'position_estimates': [],
                'velocity_estimates': [],
                'confidence_history': [],
                'track_quality': TrackQuality.INITIAL
            }
        
        track = self.tracks[target.track_id]
        track['last_update'] = time.time()
        track['position_estimates'].append((target.x, target.y, target.z, target.t))
        track['velocity_estimates'].append((target.vx, target.vy, target.vz, target.t))
        track['confidence_history'].append(detection_confidence)
        
        if len(track['position_estimates']) > 100:
            track['position_estimates'].pop(0)
            track['velocity_estimates'].pop(0)
            track['confidence_history'].pop(0)
        
        track_duration = time.time() - track['first_detection']
        avg_confidence = sum(track['confidence_history']) / len(track['confidence_history'])
        
        if track_duration > 10 and avg_confidence > 0.9:
            track['track_quality'] = TrackQuality.PRECISION
        elif track_duration > 5 and avg_confidence > 0.8:
            track['track_quality'] = TrackQuality.LOCK
        elif track_duration > 2 and avg_confidence > 0.6:
            track['track_quality'] = TrackQuality.HIGH
        elif avg_confidence > 0.4:
            track['track_quality'] = TrackQuality.CONFIRMED
        else:
            track['track_quality'] = TrackQuality.TENTATIVE
        
        target.track_quality = track['track_quality'].value
        target.classification_confidence = min(1.0, avg_confidence)
    
    def update(self, dt: float):
        if self.maintenance_time > 0:
            self.maintenance_time -= dt
            if self.maintenance_time <= 0:
                self.status = "OPERATIONAL"
                self.performance_degradation = 0.0
        
        self.current_azimuth += (360 / self.update_rate) * dt
        if self.current_azimuth >= 360:
            self.current_azimuth -= 360
        
        if self.frequency_agility and random.random() < 0.1:
            self.current_frequency = random.uniform(self.frequency_min, self.frequency_max)
        
        self.calibration_drift += random.uniform(-0.001, 0.001) * dt
        self.performance_degradation = max(0, min(0.3, abs(self.calibration_drift)))
        
        current_time = time.time()
        expired_tracks = []
        for track_id, track_data in self.tracks.items():
            if current_time - track_data['last_update'] > 30:
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del self.tracks[track_id]

class AdvancedInterceptor:
    def __init__(self, interceptor_type: InterceptorType, launch_x: float, launch_y: float, launch_z: float):
        self.type = interceptor_type
        self.capabilities = INTERCEPTOR_CAPABILITIES[interceptor_type]
        
        self.x = launch_x
        self.y = launch_y
        self.z = launch_z
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        
        self.target = None
        self.guidance_active = False
        self.fuel_remaining = 1.0
        self.flight_time = 0.0
        self.max_flight_time = 300.0
        
        self.status = "FLYING"
        self.intercept_range = 50.0
        self.proximity_fuse_range = 20.0
        
        if interceptor_type == InterceptorType.IRON_BEAM:
            self.is_laser = True
            self.beam_duration = 3.0
            self.power_remaining = 1.0
        else:
            self.is_laser = False
        
        self.navigation_error = random.gauss(0, 5)
        self.guidance_noise = random.uniform(0.95, 1.05)
        
    def update_guidance(self, target: AdvancedTarget, dt: float):
        if not self.guidance_active or not target:
            return
        
        if self.is_laser:
            self.update_laser_guidance(target, dt)
        else:
            self.update_missile_guidance(target, dt)
    
    def update_laser_guidance(self, target: AdvancedTarget, dt: float):
        if self.beam_duration <= 0:
            self.status = "EXPIRED"
            return
        
        distance = math.sqrt((target.x - self.x)**2 + (target.y - self.y)**2 + (target.z - self.z)**2)
        
        if distance <= self.capabilities["max_range"]:
            self.beam_duration -= dt
            if self.beam_duration <= 0:
                hit_probability = 0.95 * (1 - distance / self.capabilities["max_range"])
                if random.random() < hit_probability:
                    self.status = "HIT"
                else:
                    self.status = "MISS"
    
    def update_missile_guidance(self, target: AdvancedTarget, dt: float):
        if self.fuel_remaining <= 0:
            self.vz -= GRAVITY * dt
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.z += self.vz * dt
            return
        
        dx = target.x - self.x
        dy = target.y - self.y
        dz = target.z - self.z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance < self.intercept_range:
            hit_probability = self.capabilities["success_rate"] * self.guidance_noise
            if random.random() < hit_probability:
                self.status = "HIT"
            else:
                self.status = "MISS"
            return
        
        if distance < self.proximity_fuse_range and self.capabilities["warhead_type"] == "fragmentation":
            self.status = "DETONATED"
            fragment_hit_prob = 0.7 * (1 - distance / self.proximity_fuse_range)
            if random.random() < fragment_hit_prob:
                self.status = "HIT"
            return
        
        lead_time = distance / self.capabilities["max_speed"]
        predicted_x = target.x + target.vx * lead_time
        predicted_y = target.y + target.vy * lead_time
        predicted_z = target.z + target.vz * lead_time
        
        predicted_dx = predicted_x - self.x
        predicted_dy = predicted_y - self.y
        predicted_dz = predicted_z - self.z
        predicted_distance = math.sqrt(predicted_dx**2 + predicted_dy**2 + predicted_dz**2)
        
        if predicted_distance > 0:
            required_vx = predicted_dx / predicted_distance * self.capabilities["max_speed"]
            required_vy = predicted_dy / predicted_distance * self.capabilities["max_speed"]
            required_vz = predicted_dz / predicted_distance * self.capabilities["max_speed"]
            
            max_turn_rate = self.capabilities["maneuverability"] * GRAVITY / self.capabilities["max_speed"]
            
            dvx = required_vx - self.vx
            dvy = required_vy - self.vy
            dvz = required_vz - self.vz
            
            turn_magnitude = math.sqrt(dvx**2 + dvy**2 + dvz**2)
            if turn_magnitude > max_turn_rate * dt:
                scale = (max_turn_rate * dt) / turn_magnitude
                dvx *= scale
                dvy *= scale
                dvz *= scale
            
            self.vx += dvx
            self.vy += dvy
            self.vz += dvz
        
        speed = math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        if speed > self.capabilities["max_speed"]:
            scale = self.capabilities["max_speed"] / speed
            self.vx *= scale
            self.vy *= scale
            self.vz *= scale
        
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        
        self.fuel_remaining -= dt / 60.0
        self.flight_time += dt
        
        if self.flight_time > self.max_flight_time:
            self.status = "EXPIRED"

class AdvancedBattery:
    def __init__(self, x: float, y: float, z: float, battery_id: int):
        self.x = x
        self.y = y
        self.z = z
        self.battery_id = battery_id
        
        self.interceptors = {
            InterceptorType.TAMIR: 40,
            InterceptorType.DAVID_SLING: 16,
            InterceptorType.ARROW_2: 8,
            InterceptorType.ARROW_3: 4,
            InterceptorType.IRON_BEAM: 1,
            InterceptorType.PATRIOT: 12
        }
        
        self.reload_times = {itype: 0.0 for itype in InterceptorType}
        self.status = "OPERATIONAL"
        self.power_level = 1.0
        self.maintenance_time = 0.0
        
        self.radars = [
            AdvancedRadar(x, y, z + 20, RadarType.SEARCH),
            AdvancedRadar(x, y, z + 25, RadarType.TRACKING),
            AdvancedRadar(x, y, z + 30, RadarType.FIRE_CONTROL)
        ]
        
        self.active_intercepts = []
        self.engagement_doctrine = {
            ThreatPriority.IMMINENT: [InterceptorType.TAMIR, InterceptorType.DAVID_SLING],
            ThreatPriority.CRITICAL: [InterceptorType.DAVID_SLING, InterceptorType.ARROW_2],
            ThreatPriority.HIGH: [InterceptorType.TAMIR, InterceptorType.PATRIOT],
            ThreatPriority.MEDIUM: [InterceptorType.TAMIR],
            ThreatPriority.LOW: []
        }
        
        self.reaction_time = 0.5
        self.last_engagement = 0.0
        
    def can_engage(self, target: AdvancedTarget) -> Tuple[bool, Optional[InterceptorType]]:
        if self.status != "OPERATIONAL" or self.maintenance_time > 0:
            return False, None
        
        if target.intercept_attempts >= target.max_intercept_attempts:
            return False, None
        
        distance = math.sqrt((target.x - self.x)**2 + (target.y - self.y)**2 + (target.z - self.z)**2)
        
        available_types = self.engagement_doctrine.get(target.threat_priority, [])
        
        for interceptor_type in available_types:
            if (self.interceptors[interceptor_type] > 0 and 
                self.reload_times[interceptor_type] <= 0 and
                distance <= INTERCEPTOR_CAPABILITIES[interceptor_type]["max_range"] and
                distance >= INTERCEPTOR_CAPABILITIES[interceptor_type]["min_range"] and
                target.z <= INTERCEPTOR_CAPABILITIES[interceptor_type]["max_altitude"]):
                return True, interceptor_type
        
        return False, None
    
    def launch_interceptor(self, target: AdvancedTarget, interceptor_type: InterceptorType) -> AdvancedInterceptor:
        if self.interceptors[interceptor_type] <= 0:
            return None
        
        self.interceptors[interceptor_type] -= 1
        self.reload_times[interceptor_type] = INTERCEPTOR_CAPABILITIES[interceptor_type]["reload_time"]
        
        interceptor = AdvancedInterceptor(interceptor_type, self.x, self.y, self.z + 5)
        interceptor.target = target
        interceptor.guidance_active = True
        
        target.intercept_attempts += 1
        self.active_intercepts.append(interceptor)
        self.last_engagement = time.time()
        
        return interceptor
    
    def update(self, dt: float, targets: List[AdvancedTarget], terrain: TerrainModel, weather: WeatherCondition):
        if self.maintenance_time > 0:
            self.maintenance_time -= dt
            if self.maintenance_time <= 0:
                self.status = "OPERATIONAL"
        
        for interceptor_type in InterceptorType:
            if self.reload_times[interceptor_type] > 0:
                self.reload_times[interceptor_type] -= dt
                if self.reload_times[interceptor_type] <= 0:
                    if interceptor_type != InterceptorType.IRON_BEAM:
                        self.interceptors[interceptor_type] = min(
                            self.interceptors[interceptor_type] + 1,
                            40 if interceptor_type == InterceptorType.TAMIR else 16
                        )
        
        for radar in self.radars:
            radar.update(dt)
            for target in targets:
                if not target.detected:
                    can_detect, confidence = radar.can_detect(target, weather, terrain)
                    if can_detect:
                        target.detected = True
                        target.detection_time = time.time()
                        radar.update_track(target, confidence)
                elif target.track_id in radar.tracks:
                    radar.update_track(target, 0.8)
        
        active_intercepts_copy = self.active_intercepts[:]
        for interceptor in active_intercepts_copy:
            if interceptor.status == "FLYING":
                interceptor.update_guidance(interceptor.target, dt)
            else:
                self.active_intercepts.remove(interceptor)

class EnhancedDefenseSystem:
    def __init__(self):
        self.terrain = TerrainModel()
        self.atmosphere = AtmosphericModel()
        self.weather = random.choice(list(WeatherCondition))
        
        self.batteries = [
            AdvancedBattery(5000, 0, 100, 1),
            AdvancedBattery(8000, 2000, 150, 2),
            AdvancedBattery(3000, -1000, 120, 3),
            AdvancedBattery(10000, 5000, 180, 4)
        ]
        
        self.early_warning_radar = AdvancedRadar(0, 0, 200, RadarType.EARLY_WARNING)
        
        self.active_targets = []
        self.completed_engagements = []
        self.system_alerts = []
        
        self.simulation_time = 0.0
        self.engagement_log = []
        
        self.threat_level = 0
        self.total_intercepts = 0
        self.successful_intercepts = 0
        self.total_cost = 0
        
    def add_threat_wave(self, num_threats: int = None):
        if num_threats is None:
            num_threats = random.randint(1, 8)
        
        threat_types = [TargetType.ROCKET, TargetType.ARTILLERY, TargetType.MORTAR]
        if random.random() < 0.3:
            threat_types.extend([TargetType.CRUISE_MISSILE, TargetType.DRONE])
        if random.random() < 0.1:
            threat_types.extend([TargetType.BALLISTIC_MISSILE])
        if random.random() < 0.2:
            threat_types.extend([TargetType.DECOY, TargetType.CLUSTER_MUNITION])
        
        for _ in range(num_threats):
            threat_type = random.choice(threat_types)
            target = AdvancedTarget(threat_type)
            self.active_targets.append(target)
        
        print(f"Threat wave detected: {num_threats} incoming targets")
        for target in self.active_targets[-num_threats:]:
            print(f"  Track {target.track_id}: {target.type.value}")
    
    def update_system(self, dt: float):
        self.simulation_time += dt
        
        targets_to_remove = []
        for target in self.active_targets:
            target.update_physics(dt, self.atmosphere)
            
            if target.z <= 0:
                if target.predicted_impact_point:
                    impact_x, impact_y = target.predicted_impact_point
                    print(f"Track {target.track_id} IMPACT at ({int(impact_x)}, {int(impact_y)})")
                targets_to_remove.append(target)
            elif target.predicted_impact_time and target.predicted_impact_time < 0:
                targets_to_remove.append(target)
            
            if target.fragments:
                self.active_targets.extend(target.fragments)
                target.fragments = []
        
        for target in targets_to_remove:
            if target in self.active_targets:
                self.active_targets.remove(target)
        
        self.early_warning_radar.update(dt)
        
        for battery in self.batteries:
            battery.update(dt, self.active_targets, self.terrain, self.weather)
        
        self.evaluate_engagement_opportunities()
        self.update_threat_level()
    
    def evaluate_engagement_opportunities(self):
        for target in self.active_targets:
            if not target.detected:
                continue
            
            if target.threat_priority == ThreatPriority.LOW:
                continue
            
            if target.predicted_impact_time and target.predicted_impact_time > 60:
                continue
            
            best_battery = None
            best_interceptor_type = None
            best_score = 0
            
            for battery in self.batteries:
                can_engage, interceptor_type = battery.can_engage(target)
                if can_engage:
                    distance = math.sqrt((target.x - battery.x)**2 + (target.y - battery.y)**2)
                    time_to_intercept = distance / INTERCEPTOR_CAPABILITIES[interceptor_type]["max_speed"]
                    
                    if target.predicted_impact_time and time_to_intercept >= target.predicted_impact_time:
                        continue
                    
                    engagement_score = (
                        target.threat_priority.value * 10 +
                        INTERCEPTOR_CAPABILITIES[interceptor_type]["success_rate"] * 5 +
                        (1 - time_to_intercept / max(1, target.predicted_impact_time or 60)) * 3 +
                        (1 - distance / INTERCEPTOR_CAPABILITIES[interceptor_type]["max_range"]) * 2
                    )
                    
                    if engagement_score > best_score:
                        best_score = engagement_score
                        best_battery = battery
                        best_interceptor_type = interceptor_type
            
            if best_battery and best_interceptor_type:
                interceptor = best_battery.launch_interceptor(target, best_interceptor_type)
                if interceptor:
                    self.total_intercepts += 1
                    self.total_cost += INTERCEPTOR_CAPABILITIES[best_interceptor_type]["cost"]
                    
                    print(f"ENGAGEMENT: Track {target.track_id} ({target.type.value})")
                    print(f"  Threat Priority: {target.threat_priority.name}")
                    print(f"  Battery {best_battery.battery_id} firing {best_interceptor_type.value}")
                    print(f"  Target at ({int(target.x)}, {int(target.y)}, {int(target.z)})")
                    print(f"  Impact prediction: {target.predicted_impact_time:.1f}s")
                    print(f"  Track quality: {target.track_quality:.2f}")
                    
                    engagement_data = {
                        'time': self.simulation_time,
                        'target_id': target.track_id,
                        'target_type': target.type.value,
                        'threat_priority': target.threat_priority.name,
                        'interceptor_type': best_interceptor_type.value,
                        'battery_id': best_battery.battery_id,
                        'cost': INTERCEPTOR_CAPABILITIES[best_interceptor_type]["cost"]
                    }
                    self.engagement_log.append(engagement_data)
    
    def update_threat_level(self):
        high_priority_threats = sum(1 for target in self.active_targets 
                                   if target.threat_priority.value >= 3)
        self.threat_level = min(5, high_priority_threats)
    
    def run_simulation(self, duration: float = 300.0, threat_interval: float = 30.0):
        dt = 0.1
        last_threat_time = 0.0
        
        print("Enhanced Missile Defense Simulation Starting")
        print(f"Weather: {self.weather.value}")
        print(f"Wind: {self.atmosphere.wind_speed:.1f} m/s from {self.atmosphere.wind_direction:.0f}°")
        print("=" * 60)
        
        while self.simulation_time < duration:
            if self.simulation_time - last_threat_time >= threat_interval:
                self.add_threat_wave()
                last_threat_time = self.simulation_time
            
            self.update_system(dt)
            
            completed_intercepts = []
            for battery in self.batteries:
                for interceptor in battery.active_intercepts:
                    if interceptor.status in ["HIT", "MISS", "DETONATED", "EXPIRED"]:
                        completed_intercepts.append(interceptor)
                        if interceptor.status == "HIT":
                            self.successful_intercepts += 1
                            print(f"INTERCEPT SUCCESS: {interceptor.type.value} destroyed target")
                        else:
                            print(f"INTERCEPT FAILED: {interceptor.type.value} - {interceptor.status}")
            
            time.sleep(0.01)
        
        self.print_final_report()
    
    def print_final_report(self):
        print("\n" + "=" * 60)
        print("FINAL SIMULATION REPORT")
        print("=" * 60)
        print(f"Total simulation time: {self.simulation_time:.1f} seconds")
        print(f"Weather conditions: {self.weather.value}")
        print(f"Total threats detected: {len(self.engagement_log) + len(self.active_targets)}")
        print(f"Total intercept attempts: {self.total_intercepts}")
        print(f"Successful intercepts: {self.successful_intercepts}")
        if self.total_intercepts > 0:
            success_rate = (self.successful_intercepts / self.total_intercepts) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print(f"Total cost: ${self.total_cost:,}")
        
        print("\nBattery Status:")
        for battery in self.batteries:
            print(f"  Battery {battery.battery_id} ({battery.status}):")
            for itype, count in battery.interceptors.items():
                print(f"    {itype.value}: {count} remaining")
        
        print("\nThreat Type Analysis:")
        threat_types = {}
        for engagement in self.engagement_log:
            ttype = engagement['target_type']
            threat_types[ttype] = threat_types.get(ttype, 0) + 1
        
        for ttype, count in threat_types.items():
            print(f"  {ttype}: {count} engagements")
        
        print("\nInterceptor Usage:")
        interceptor_usage = {}
        interceptor_costs = {}
        for engagement in self.engagement_log:
            itype = engagement['interceptor_type']
            interceptor_usage[itype] = interceptor_usage.get(itype, 0) + 1
            interceptor_costs[itype] = interceptor_costs.get(itype, 0) + engagement['cost']
        
        for itype, count in interceptor_usage.items():
            cost = interceptor_costs[itype]
            print(f"  {itype}: {count} fired, ${cost:,} total cost")

def create_complex_scenario():
    """Create a complex multi-wave attack scenario"""
    system = EnhancedDefenseSystem()
    
    print("SCENARIO: Multi-wave coordinated attack")
    wave1_threats = []
    for _ in range(12):
        if random.random() < 0.7:
            target = AdvancedTarget(TargetType.ROCKET)
        else:
            target = AdvancedTarget(TargetType.ARTILLERY)
        wave1_threats.append(target)
    
    system.active_targets.extend(wave1_threats)
    print(f"Wave 1: {len(wave1_threats)} rockets and artillery shells")
    
    for i in range(600):  
        system.update_system(0.1)
        time.sleep(0.001)
        
        if i == 300:
            wave2_threats = []
            for _ in range(4):
                target = AdvancedTarget(TargetType.CRUISE_MISSILE)
                wave2_threats.append(target)
            for _ in range(6):
                target = AdvancedTarget(TargetType.DRONE)
                wave2_threats.append(target)
            
            system.active_targets.extend(wave2_threats)
            print(f"Wave 2: {len(wave2_threats)} cruise missiles and drones")
        
        if i == 450:
            wave3_threats = []
       
            for _ in range(2):
                target = AdvancedTarget(TargetType.BALLISTIC_MISSILE)
                wave3_threats.append(target)
            for _ in range(5):
                target = AdvancedTarget(TargetType.DECOY)
                wave3_threats.append(target)
            
            system.active_targets.extend(wave3_threats)
            print(f"Wave 3: {len(wave3_threats)} ballistic missiles with decoys")
    
    system.print_final_report()
    return system

def create_saturation_scenario():
    """Create a saturation attack to test system limits"""
    system = EnhancedDefenseSystem()
    system.weather = WeatherCondition.HEAVY_RAIN  
    
    print("SCENARIO: Saturation attack in adverse weather")
    print(f"Weather: {system.weather.value}")
    
    saturation_threats = []
    for _ in range(50):  
        threat_types = [TargetType.ROCKET, TargetType.MORTAR, TargetType.ARTILLERY]
        target = AdvancedTarget(random.choice(threat_types))
        
        target.t = random.uniform(0, 5)
        saturation_threats.append(target)
    
    system.active_targets.extend(saturation_threats)
    print(f"Saturation wave: {len(saturation_threats)} simultaneous threats")
    
    for i in range(1200):  
        system.update_system(0.1)
        time.sleep(0.001)
    
    system.print_final_report()
    return system

def create_stealth_scenario():
    """Create a scenario with advanced stealth threats"""
    system = EnhancedDefenseSystem()
    system.weather = WeatherCondition.FOG  
    
    print("SCENARIO: Stealth and electronic warfare")
    print(f"Weather: {system.weather.value}")
    
    stealth_threats = []
    
    for _ in range(6):
        target = AdvancedTarget(TargetType.CRUISE_MISSILE)
        target.ecm_active = True
        target.ecm_types = [ECMType.STEALTH, ECMType.JAMMING]
        stealth_threats.append(target)
    
    for _ in range(10):
        target = AdvancedTarget(TargetType.DRONE)
        target.ecm_active = True
        target.ecm_types = [ECMType.STEALTH]
        target.base_rcs *= 0.1
        stealth_threats.append(target)
    
    for _ in range(8):
        target = AdvancedTarget(TargetType.DECOY)
        target.ecm_active = True
        target.ecm_types = [ECMType.CHAFF, ECMType.DECOY]
        target.base_rcs *= 5
        stealth_threats.append(target)
    
    system.active_targets.extend(stealth_threats)
    print(f"Stealth wave: {len(stealth_threats)} low-observable threats")
    
    for i in range(1800):
        system.update_system(0.1)
        time.sleep(0.001)
    
    system.print_final_report()
    return system

def analyze_system_performance(scenarios_results):
    """Analyze performance across multiple scenarios"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    total_intercepts = sum(s.total_intercepts for s in scenarios_results)
    total_successful = sum(s.successful_intercepts for s in scenarios_results)
    total_cost = sum(s.total_cost for s in scenarios_results)
    
    if total_intercepts > 0:
        overall_success_rate = (total_successful / total_intercepts) * 100
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    
    print(f"Total Intercepts Attempted: {total_intercepts}")
    print(f"Total Successful Intercepts: {total_successful}")
    print(f"Total System Cost: ${total_cost:,}")
    
    if total_successful > 0:
        cost_per_success = total_cost / total_successful
        print(f"Cost per Successful Intercept: ${cost_per_success:,.0f}")
    
    print("\nScenario Breakdown:")
    scenario_names = ["Complex Multi-wave", "Saturation Attack", "Stealth & ECM"]
    for i, (name, system) in enumerate(zip(scenario_names, scenarios_results)):
        if system.total_intercepts > 0:
            success_rate = (system.successful_intercepts / system.total_intercepts) * 100
            print(f"  {name}: {success_rate:.1f}% success rate")
        else:
            print(f"  {name}: No intercepts attempted")

def run_comprehensive_test():
    """Run all test scenarios and analyze results"""
    print("ENHANCED MISSILE DEFENSE COMPREHENSIVE TEST")
    print("=" * 80)
    print("Testing system performance across multiple threat scenarios...")
    print()
    
    scenarios = []
    
    print("Running Scenario 1 of 3...")
    scenarios.append(create_complex_scenario())
    
    print("\nRunning Scenario 2 of 3...")
    scenarios.append(create_saturation_scenario())
    
    print("\nRunning Scenario 3 of 3...")
    scenarios.append(create_stealth_scenario())
    
    analyze_system_performance(scenarios)

if __name__ == "__main__":
    run_comprehensive_test()
