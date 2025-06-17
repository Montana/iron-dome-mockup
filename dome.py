import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum
from datetime import datetime

class TargetType(Enum):
    ROCKET = "rocket"
    MORTAR = "mortar"
    ARTILLERY = "artillery"
    MISSILE = "missile"

class RadarType(Enum):
    SEARCH = "search"
    TRACKING = "tracking"

class InterceptorType(Enum):
    TAMIR = "tamir"
    DAVID_SLING = "david_sling"
    ARROW = "arrow"

class WeatherCondition(Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"

class ThreatPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SystemMode(Enum):
    NORMAL = "normal"
    HIGH_ALERT = "high_alert"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"

GRAVITY = 9.81
CITY_MIN = 2000
CITY_MAX = 8000
RADAR_INTERVAL = 0.5
INTERCEPTOR_SPEED = 1000
RESPONSE_DELAY = 1.0
ENERGY_THRESHOLD = 120000
RADAR_DETECTION_RANGE = 20000
RADAR_DETECTION_PROBABILITY = 0.95
INTERCEPTOR_SUCCESS_BASE = 0.85
BATTERY_RELOAD_TIME = 15
MAX_INTERCEPTORS_PER_BATTERY = 20
WEATHER_DEGRADATION = 0.1
SYSTEM_LATENCY = 0.2
TRACK_DEGRADATION_RATE = 0.05
MIN_TRACK_QUALITY = 0.3
CRITICAL_IMPACT_THRESHOLD = 5000
HIGH_IMPACT_THRESHOLD = 7000
MEDIUM_IMPACT_THRESHOLD = 9000

TARGET_RCS = {
    TargetType.ROCKET: 0.1,
    TargetType.MORTAR: 0.05,
    TargetType.ARTILLERY: 0.15,
    TargetType.MISSILE: 0.2
}

TARGET_PRIORITY = {
    TargetType.ROCKET: 1,
    TargetType.MORTAR: 2,
    TargetType.ARTILLERY: 3,
    TargetType.MISSILE: 4
}

INTERCEPTOR_CAPABILITIES = {
    InterceptorType.TAMIR: {
        "max_range": 5000,
        "max_altitude": 5000,
        "success_rate": 0.85,
        "reload_time": 15,
        "cost": 1
    },
    InterceptorType.DAVID_SLING: {
        "max_range": 10000,
        "max_altitude": 10000,
        "success_rate": 0.90,
        "reload_time": 30,
        "cost": 2
    },
    InterceptorType.ARROW: {
        "max_range": 15000,
        "max_altitude": 15000,
        "success_rate": 0.95,
        "reload_time": 45,
        "cost": 3
    }
}

WEATHER_EFFECTS = {
    WeatherCondition.CLEAR: 0.0,
    WeatherCondition.CLOUDY: 0.2,
    WeatherCondition.RAINY: 0.4,
    WeatherCondition.STORMY: 0.6
}

@dataclass
class CurrentState:
    timestamp: datetime
    weather: WeatherCondition
    active_tracks: Dict[int, 'Target']
    battery_status: Dict[int, Dict[InterceptorType, int]]
    system_status: str
    threat_level: int
    total_intercepts: int
    successful_intercepts: int
    intercept_cost: int = 0
    max_intercepts_per_engagement: int = 3
    curr_mode: SystemMode = SystemMode.NORMAL
    curr_resources: Dict[str, float] = None
    curr_alerts: List[str] = None
    curr_maintenance: Dict[int, float] = None
    curr_performance: Dict[str, float] = None

    def __post_init__(self):
        if self.curr_resources is None:
            self.curr_resources = {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "network_load": 0.0
            }
        if self.curr_alerts is None:
            self.curr_alerts = []
        if self.curr_maintenance is None:
            self.curr_maintenance = {}
        if self.curr_performance is None:
            self.curr_performance = {
                "detection_rate": 1.0,
                "track_quality": 1.0,
                "response_time": 0.0
            }

    def update(self, dt: float):
        self.timestamp = datetime.now()
        self.threat_level = min(5, len(self.active_tracks))
        
        if self.threat_level >= 4:
            self.curr_mode = SystemMode.CRITICAL
        elif self.threat_level >= 2:
            self.curr_mode = SystemMode.HIGH_ALERT
        else:
            self.curr_mode = SystemMode.NORMAL

        self.curr_resources["cpu_usage"] = min(1.0, 0.3 + (len(self.active_tracks) * 0.1))
        self.curr_resources["memory_usage"] = min(1.0, 0.2 + (len(self.active_tracks) * 0.05))
        self.curr_resources["network_load"] = min(1.0, 0.4 + (len(self.active_tracks) * 0.15))

        self.curr_performance["detection_rate"] = max(0.7, 1.0 - (WEATHER_EFFECTS[self.weather] * 0.3))
        self.curr_performance["track_quality"] = max(0.8, 1.0 - (len(self.active_tracks) * 0.05))
        self.curr_performance["response_time"] = SYSTEM_LATENCY * (1 + len(self.active_tracks) * 0.1)

        if self.curr_resources["cpu_usage"] > 0.9:
            self.curr_alerts.append("High CPU usage")
        if self.curr_resources["memory_usage"] > 0.9:
            self.curr_alerts.append("High memory usage")
        if self.curr_resources["network_load"] > 0.9:
            self.curr_alerts.append("High network load")

        for battery_id in self.curr_maintenance:
            self.curr_maintenance[battery_id] = max(0, self.curr_maintenance[battery_id] - dt)
            if self.curr_maintenance[battery_id] == 0:
                del self.curr_maintenance[battery_id]

    def can_intercept_more(self) -> bool:
        return (self.total_intercepts < self.max_intercepts_per_engagement and 
                self.curr_mode != SystemMode.MAINTENANCE)

    def get_system_status(self) -> str:
        if self.curr_mode == SystemMode.CRITICAL:
            return "CRITICAL"
        elif self.curr_mode == SystemMode.HIGH_ALERT:
            return "HIGH_ALERT"
        elif self.curr_mode == SystemMode.MAINTENANCE:
            return "MAINTENANCE"
        else:
            return "OPERATIONAL"

@dataclass
class Radar:
    x: float
    radar_type: RadarType
    detection_range: float
    track_capacity: int
    curr_tracks: Dict[int, float] = None
    curr_status: str = "OPERATIONAL"
    curr_performance: float = 1.0
    curr_maintenance: float = 0.0

    def __post_init__(self):
        if self.curr_tracks is None:
            self.curr_tracks = {}

    def can_detect(self, target: 'Target', weather_condition: WeatherCondition) -> bool:
        if self.curr_status != "OPERATIONAL" or self.curr_maintenance > 0:
            return False
            
        distance = abs(target.x - self.x)
        if distance > self.detection_range:
            return False
        
        rcs = TARGET_RCS[target.type]
        weather_factor = 1 - (WEATHER_DEGRADATION * WEATHER_EFFECTS[weather_condition])
        detection_chance = (RADAR_DETECTION_PROBABILITY * 
                          (1 - distance/self.detection_range) * 
                          rcs * 
                          weather_factor *
                          self.curr_performance)
        return random.random() < detection_chance

    def update(self, dt: float):
        if self.curr_maintenance > 0:
            self.curr_maintenance = max(0, self.curr_maintenance - dt)
            if self.curr_maintenance == 0:
                self.curr_status = "OPERATIONAL"
                self.curr_performance = 1.0

@dataclass
class Battery:
    x: float
    coverage_min: float
    coverage_max: float
    interceptors: Dict[InterceptorType, int]
    radars: List[Radar]
    reload_times: Dict[InterceptorType, float] = None
    curr_status: str = "OPERATIONAL"
    battery_id: int = 0
    curr_performance: float = 1.0
    curr_maintenance: float = 0.0

    def __post_init__(self):
        if self.reload_times is None:
            self.reload_times = {itype: 0 for itype in InterceptorType}

    def can_engage(self, target: 'Target') -> Tuple[bool, InterceptorType]:
        if self.curr_status != "OPERATIONAL" or self.curr_maintenance > 0:
            return False, None

        if not (self.coverage_min <= target.x <= self.coverage_max):
            return False, None

        for itype, count in self.interceptors.items():
            if count > 0 and self.reload_times[itype] <= 0:
                capabilities = INTERCEPTOR_CAPABILITIES[itype]
                if (abs(target.x - self.x) <= capabilities["max_range"] and
                    target.y <= capabilities["max_altitude"]):
                    return True, itype
        return False, None

    def fire(self, interceptor_type: InterceptorType):
        self.interceptors[interceptor_type] -= 1
        if self.interceptors[interceptor_type] == 0:
            self.reload_times[interceptor_type] = INTERCEPTOR_CAPABILITIES[interceptor_type]["reload_time"]

    def update(self, dt: float):
        if self.curr_maintenance > 0:
            self.curr_maintenance = max(0, self.curr_maintenance - dt)
            if self.curr_maintenance == 0:
                self.curr_status = "OPERATIONAL"
                self.curr_performance = 1.0

        for itype in InterceptorType:
            if self.reload_times[itype] > 0:
                self.reload_times[itype] = max(0, self.reload_times[itype] - dt)
                if self.reload_times[itype] == 0:
                    self.interceptors[itype] = MAX_INTERCEPTORS_PER_BATTERY

class Target:
    def __init__(self):
        self.x0 = random.uniform(-15000, 15000)
        self.alt = 0
        self.speed = random.uniform(400, 600)
        self.angle = math.radians(random.uniform(35, 75))
        self.mass = random.uniform(40, 100)
        self.vx = self.speed * math.cos(self.angle)
        self.vy = self.speed * math.sin(self.angle)
        self.x = self.x0
        self.y = self.alt
        self.t = 0
        self.detected = False
        self.track_quality = 0.0
        self.type = random.choice(list(TargetType))
        self.priority = random.uniform(0, 1)
        self.track_id = random.randint(1000, 9999)
        self.predicted_impact_point = None
        self.predicted_impact_time = None
        self.curr_status = "TRACKING"
        self.detection_time = None
        self.threat_priority = None
        self.intercept_decision = False
        self.curr_track_quality = 1.0
        self.curr_velocity = (self.vx, self.vy)
        self.curr_position = (self.x, self.y)

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt - 0.5 * GRAVITY * dt**2
        self.vy -= GRAVITY * dt
        self.t += dt
        
        self.curr_position = (self.x, self.y)
        self.curr_velocity = (self.vx, self.vy)
        
        if self.detected:
            self.track_quality = max(MIN_TRACK_QUALITY, 
                                   self.track_quality - TRACK_DEGRADATION_RATE * dt)
            self.curr_track_quality = self.track_quality
            self.update_prediction()
            if self.track_quality < MIN_TRACK_QUALITY:
                self.curr_status = "LOST_TRACK"

    def update_prediction(self):
        if self.y <= 0:
            self.predicted_impact_point = self.x
            self.predicted_impact_time = 0
            return

        vy = self.vy
        y = self.y
        t = 0
        while y > 0:
            y += vy * RADAR_INTERVAL - 0.5 * GRAVITY * RADAR_INTERVAL**2
            vy -= GRAVITY * RADAR_INTERVAL
            t += RADAR_INTERVAL

        self.predicted_impact_point = self.x + self.vx * t
        self.predicted_impact_time = t

    def assess_threat(self):
        base_priority = TARGET_PRIORITY[self.type]
        impact_distance = abs(self.predicted_impact_point - CITY_MIN)
        
        if impact_distance < CRITICAL_IMPACT_THRESHOLD:
            self.threat_priority = ThreatPriority.CRITICAL
        elif impact_distance < HIGH_IMPACT_THRESHOLD:
            self.threat_priority = ThreatPriority.HIGH
        elif impact_distance < MEDIUM_IMPACT_THRESHOLD:
            self.threat_priority = ThreatPriority.MEDIUM
        else:
            self.threat_priority = ThreatPriority.LOW

    def energy(self):
        return 0.5 * self.mass * (self.vx**2 + self.vy**2)

    def distance_to_battery(self, battery_x: float) -> float:
        return abs(self.x - battery_x)

def calculate_intercept_probability(target: Target, interceptor_type: InterceptorType, 
                                  track_quality: float, weather_condition: WeatherCondition) -> float:
    base_prob = INTERCEPTOR_CAPABILITIES[interceptor_type]["success_rate"]
    speed_factor = 1 - (target.speed - 400) / 400
    track_factor = track_quality
    weather_factor = 1 - (WEATHER_DEGRADATION * WEATHER_EFFECTS[weather_condition])
    return base_prob * speed_factor * track_factor * weather_factor

def assess_engagement(target: Target, current_state: CurrentState) -> bool:
    if not current_state.can_intercept_more():
        return False

    target.assess_threat()
    
    if target.threat_priority == ThreatPriority.CRITICAL:
        return True
    
    if target.threat_priority == ThreatPriority.HIGH:
        return random.random() < 0.9
    
    if target.threat_priority == ThreatPriority.MEDIUM:
        return random.random() < 0.7
    
    if target.threat_priority == ThreatPriority.LOW:
        return random.random() < 0.4
    
    return False

def simulate_engagement():
    current_state = CurrentState(
        timestamp=datetime.now(),
        weather=random.choice(list(WeatherCondition)),
        active_tracks={},
        battery_status={},
        system_status="OPERATIONAL",
        threat_level=0,
        total_intercepts=0,
        successful_intercepts=0
    )
    
    batteries = [
        Battery(
            x=2000,
            coverage_min=0,
            coverage_max=5000,
            interceptors={itype: MAX_INTERCEPTORS_PER_BATTERY for itype in InterceptorType},
            radars=[
                Radar(2000, RadarType.SEARCH, 25000, 100),
                Radar(2000, RadarType.TRACKING, 15000, 50)
            ],
            battery_id=1
        ),
        Battery(
            x=5000,
            coverage_min=3000,
            coverage_max=7000,
            interceptors={itype: MAX_INTERCEPTORS_PER_BATTERY for itype in InterceptorType},
            radars=[
                Radar(5000, RadarType.SEARCH, 25000, 100),
                Radar(5000, RadarType.TRACKING, 15000, 50)
            ],
            battery_id=2
        ),
        Battery(
            x=8000,
            coverage_min=6000,
            coverage_max=10000,
            interceptors={itype: MAX_INTERCEPTORS_PER_BATTERY for itype in InterceptorType},
            radars=[
                Radar(8000, RadarType.SEARCH, 25000, 100),
                Radar(8000, RadarType.TRACKING, 15000, 50)
            ],
            battery_id=3
        )
    ]
    
    for _ in range(6):
        target = Target()
        engaged = False
        while target.y >= 0:
            target.update(RADAR_INTERVAL)
            time.sleep(SYSTEM_LATENCY)
            
            for battery in batteries:
                battery.update(RADAR_INTERVAL)
                current_state.battery_status[battery.battery_id] = battery.interceptors.copy()
            
            if not target.detected:
                for battery in batteries:
                    for radar in battery.radars:
                        if radar.can_detect(target, current_state.weather):
                            target.detected = True
                            target.track_quality = 1.0
                            target.detection_time = current_state.timestamp
                            current_state.active_tracks[target.track_id] = target
                            print(f"Track {target.track_id} | New {target.type.value} detected at {round(target.t, 2)}s")
                            break
                    if target.detected:
                        break
            
            if target.detected and CITY_MIN <= target.x <= CITY_MAX and target.energy() > ENERGY_THRESHOLD:
                target.update_prediction()
                
                if assess_engagement(target, current_state):
                    best_battery = None
                    best_interceptor = None
                    best_intercept_time = float('inf')
                    
                    for battery in batteries:
                        can_engage, interceptor_type = battery.can_engage(target)
                        if can_engage:
                            dx = target.distance_to_battery(battery.x)
                            t_intercept = dx / INTERCEPTOR_SPEED + RESPONSE_DELAY
                            if t_intercept < best_intercept_time:
                                best_intercept_time = t_intercept
                                best_battery = battery
                                best_interceptor = interceptor_type
                    
                    if best_battery and target.predicted_impact_time > best_intercept_time:
                        intercept_prob = calculate_intercept_probability(
                            target, best_interceptor, target.track_quality, current_state.weather
                        )
                        
                        print(f"Track {target.track_id} | {target.type.value} at {int(target.x)}m alt {int(target.y)}m")
                        print(f"Threat Priority: {target.threat_priority.name}")
                        print(f"Predicted impact: {int(target.predicted_impact_point)}m in {round(target.predicted_impact_time,1)}s")
                        print(f"Interceptor {best_interceptor.value} fired from battery {best_battery.battery_id}")
                        print(f"ETA {round(target.predicted_impact_time,1)}s vs response {round(best_intercept_time,1)}s")
                        print(f"Intercept probability: {intercept_prob:.1%}")
                        print(f"Current weather: {current_state.weather.value}")
                        print(f"System status: {current_state.get_system_status()}")
                        print(f"Current performance: {current_state.curr_performance}")
                        print(f"Current resources: {current_state.curr_resources}")
                        if current_state.curr_alerts:
                            print(f"Current alerts: {current_state.curr_alerts}")
                        
                        best_battery.fire(best_interceptor)
                        engaged = True
                        current_state.total_intercepts += 1
                        current_state.intercept_cost += INTERCEPTOR_CAPABILITIES[best_interceptor]["cost"]
                        
                        if random.random() < intercept_prob:
                            print("Target intercepted!")
                            current_state.successful_intercepts += 1
                        else:
                            print("Intercept failed!")
                        break
                else:
                    print(f"Track {target.track_id} | Threat assessed as {target.threat_priority.name} - No intercept")
                    break
                if target.detected:
                    print(f"Track {target.track_id} | Threat detected but no time to intercept")
                    break
            
            current_state.update(RADAR_INTERVAL)
            time.sleep(0.05)
        
        if not engaged:
            print("No launch")
        print("-" * 50)
        print(f"Current system status: {current_state.get_system_status()}")
        print(f"Active tracks: {len(current_state.active_tracks)}")
        print(f"Success rate: {current_state.successful_intercepts}/{current_state.total_intercepts}")
        print(f"Total intercept cost: {current_state.intercept_cost}")
        print(f"Current performance: {current_state.curr_performance}")
        print(f"Current resources: {current_state.curr_resources}")
        if current_state.curr_alerts:
            print(f"Current alerts: {current_state.curr_alerts}")
        print("-" * 50)

if __name__ == "__main__":
    simulate_engagement()

