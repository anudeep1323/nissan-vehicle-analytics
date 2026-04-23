import json
import random
import time
import math
from datetime import datetime
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Each VIN has its own baseline — some vehicles run hotter, older ones degrade faster
class Vehicle:
    def __init__(self, vin):
        self.vin = vin
        self.mileage = random.randint(5000, 180000)
        self.age_years = random.uniform(0.5, 12)
        # Vehicle-specific baselines
        self.base_engine_temp = random.gauss(195, 8)
        self.base_battery = random.gauss(13.4, 0.3)
        self.base_oil_pressure = random.gauss(48, 5)
        self.brake_pad_mm = random.uniform(2, 12)
        self.transmission_health = random.uniform(0.7, 1.0)
        # Degradation state — gradually worsens over time
        self.degradation = min(1.0, self.mileage / 200000)
        self.is_degrading = random.random() < (0.15 + self.degradation * 0.2)
        self.degradation_rate = random.uniform(0.0001, 0.001) if self.is_degrading else 0
        # Current state
        self.speed = random.uniform(0, 80)
        self.rpm = 800

    def update(self):
        # Simulate realistic driving behavior
        speed_change = random.gauss(0, 5)
        self.speed = max(0, min(120, self.speed + speed_change))
        target_rpm = 800 + self.speed * 45 + random.gauss(0, 200)
        self.rpm = max(600, min(6500, self.rpm * 0.8 + target_rpm * 0.2))

        # Mileage increases
        self.mileage += self.speed / 3600  # per second at current speed

        # Gradual degradation
        self.degradation = min(1.0, self.degradation + self.degradation_rate)
        self.brake_pad_mm = max(0.5, self.brake_pad_mm - random.uniform(0, 0.0001))

        # Engine temp — correlated with RPM, speed, degradation
        engine_heat = (self.rpm / 6500) * 30 + (self.speed / 120) * 15
        degradation_heat = self.degradation * 25
        self.engine_temp = self.base_engine_temp + engine_heat + degradation_heat + random.gauss(0, 3)

        # Battery — degrades with age
        age_factor = self.age_years / 12
        self.battery = self.base_battery - age_factor * 0.8 - self.degradation * 0.5 + random.gauss(0, 0.15)
        self.battery = max(9.0, min(14.8, self.battery))

        # Oil pressure — drops when engine is hot and degraded
        heat_factor = max(0, (self.engine_temp - 200) / 100)
        self.oil_pressure = self.base_oil_pressure - heat_factor * 15 - self.degradation * 10 + random.gauss(0, 3)
        self.oil_pressure = max(5, min(90, self.oil_pressure))

        # Transmission temp — correlated with engine temp
        self.transmission_temp = self.engine_temp * 0.45 + random.gauss(0, 5)

        # Check engine — probabilistic based on actual conditions
        check_engine = (
            self.engine_temp > 250 or
            self.battery < 11.0 or
            self.oil_pressure < 15 or
            self.brake_pad_mm < 1.5 or
            (self.degradation > 0.85 and random.random() < 0.03)
        )

        return {
            "vin": self.vin,
            "timestamp": datetime.utcnow().isoformat(),
            "speed_mph": round(self.speed, 2),
            "engine_temp_c": round(self.engine_temp, 2),
            "battery_voltage": round(self.battery, 2),
            "oil_pressure_psi": round(self.oil_pressure, 2),
            "rpm": round(self.rpm, 0),
            "fuel_level_pct": round(random.uniform(5, 100), 2),
            "tire_pressure_fl": round(random.gauss(32, 1.5), 1),
            "tire_pressure_fr": round(random.gauss(32, 1.5), 1),
            "tire_pressure_rl": round(random.gauss(32, 1.5), 1),
            "tire_pressure_rr": round(random.gauss(32, 1.5), 1),
            "brake_pad_mm": round(self.brake_pad_mm, 2),
            "transmission_temp_c": round(self.transmission_temp, 2),
            "check_engine_light": check_engine,
            "mileage": int(self.mileage)
        }

# Initialize 100 vehicles with persistent state
VEHICLES = {f"VIN{str(i).zfill(6)}": Vehicle(f"VIN{str(i).zfill(6)}") for i in range(100)}

def run(events_per_sec=500):
    print(f"Producing {events_per_sec} events/sec with realistic vehicle physics...")
    vins = list(VEHICLES.keys())
    while True:
        batch_start = time.time()
        for _ in range(events_per_sec):
            vin = random.choice(vins)
            vehicle = VEHICLES[vin]
            event = vehicle.update()
            producer.send('vehicle-telemetry', key=vin.encode(), value=event)
        producer.flush()
        elapsed = time.time() - batch_start
        time.sleep(max(0, 1.0 - elapsed))
        print(f"Sent {events_per_sec} events | sample: VIN000000 engine={VEHICLES['VIN000000'].engine_temp:.1f}C battery={VEHICLES['VIN000000'].battery:.2f}V degradation={VEHICLES['VIN000000'].degradation:.3f}")

if __name__ == "__main__":
    run()