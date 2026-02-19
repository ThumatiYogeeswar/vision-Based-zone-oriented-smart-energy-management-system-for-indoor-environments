import cv2
import time
import requests
import RPi.GPIO as GPIO
from ultralytics import YOLO

# ================== BLYNK ==================
BLYNK_TOKEN   = "hUADR7gDUVr3Rt3hTC70XcLbSs_8BGi2"

VPIN_REFRESH   = "V0"
VPIN_LCD_L1    = "V1"
VPIN_LCD_L2    = "V2"
VPIN_SELECT    = "V6"
VPIN_ZONE_TEXT = "V12"
VPIN_ZONE_STATUS = "V13"
VPIN_ENERGY    = "V21"
VPIN_ZONE_DATA = "V11"

BLYNK_GET    = "https://blynk.cloud/external/api/get"
BLYNK_UPDATE = "https://blynk.cloud/external/api/update"
session = requests.Session()

# ================== CAMERA SCALE ==================
WEB_WIDTH  = 475
WEB_HEIGHT = 500

# ================== GPIO ==================
GPIO.setmode(GPIO.BCM)
gpio_state = {}
zones = {}

# ================== LOAD ZONES ==================
def load_zones(cam_w, cam_h):
    try:
        data = session.get(BLYNK_GET, params={"token": BLYNK_TOKEN, "pin": VPIN_ZONE_DATA}, timeout=2).json()
    except:
        return {}
    sx = cam_w / WEB_WIDTH
    sy = cam_h / WEB_HEIGHT
    loaded = {}
    for z in data.get("zones", []):
        try:
            zid = int(z["zone"].replace("Zone", ""))
            gpio = int(z["gpio"])
            p = z["position"]
            loaded[zid] = {
                "x1": int(p["x"] * sx),
                "y1": int(p["y"] * sy),
                "x2": int((p["x"] + p["width"]) * sx),
                "y2": int((p["y"] + p["height"]) * sy),
                "gpio": gpio
            }
        except:
            continue
    return loaded

# ================== GPIO SETUP ==================
def setup_gpio(zones_dict):
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    gpio_state.clear()
    for zid in zones_dict:
        pin = zones_dict[zid]["gpio"]
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)
        gpio_state[zid] = False

# ================== RELAY ==================
def set_relay(zone, on):
    pin = zones[zone]["gpio"]
    GPIO.output(pin, GPIO.LOW if on else GPIO.HIGH)

# ================== BLYNK HELPERS ==================
def blynk_read(pin):
    try:
        return int(session.get(BLYNK_GET, params={"token": BLYNK_TOKEN, "pin": pin}, timeout=1).text)
    except:
        return 0

def blynk_write(pin, value):
    try:
        session.get(BLYNK_UPDATE, params={"token": BLYNK_TOKEN, "pin": pin, "value": value}, timeout=1)
    except:
        pass

def lcd_write(l1="", l2=""):
    blynk_write(VPIN_LCD_L1, l1[:16])
    blynk_write(VPIN_LCD_L2, l2[:16])

# ================== CONSTANTS ==================
FAN_POWER_KW = 0.075
LEAVE_CONFIRM_TIME = 1.5
MIN_BOX_AREA = 1000   # Lower for small objects
CONFIDENCE = 0.3      # Lower for better detection

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
time.sleep(1)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Camera not detected")
h, w = frame.shape[:2]

# ================== LOAD INITIAL ZONES ==================
zones = load_zones(w, h)
setup_gpio(zones)
zone_ids = sorted(zones)

# ================== YOLO ==================
model = YOLO("best.pt")  # Make sure best.pt is your trained model

# ================== STATE ==================
zone_state = {z: False for z in zone_ids}
prev_zone_state = zone_state.copy()
last_seen_time = {z: 0.0 for z in zone_ids}
zone_active_time = {z: 0.0 for z in zone_ids}
zone_energy = {z: 0.0 for z in zone_ids}

program_start = time.time()
last_update = time.time()
last_lcd = 0
last_refresh = 0
last_energy_push = 0

lcd_write("System Ready", "Dynamic Zones")

# ================== MAIN LOOP ==================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()

        # -------- REFRESH BUTTON --------
        if blynk_read(VPIN_REFRESH) == 1 and now - last_refresh > 2:
            zones = load_zones(w, h)
            setup_gpio(zones)
            zone_ids = sorted(zones)
            zone_state = {z: False for z in zone_ids}
            prev_zone_state = zone_state.copy()
            last_seen_time = {z: 0.0 for z in zone_ids}
            zone_active_time = {z: 0.0 for z in zone_ids}
            zone_energy = {z: 0.0 for z in zone_ids}
            last_refresh = now
            lcd_write("Zones Reloaded", f"{len(zone_ids)} zones")

        # -------- YOLO DETECTION (Fixed) --------
        results = model.predict(frame, conf=CONFIDENCE, verbose=False)
        detected_now = set()
        zone_count = {z: 0 for z in zone_ids}

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BOX_AREA:
                continue
            # Check overlap with zones
            for zid in zone_ids:
                z = zones[zid]
                if not (x2 < z["x1"] or x1 > z["x2"] or y2 < z["y1"] or y1 > z["y2"]):
                    detected_now.add(zid)
                    zone_count[zid] += 1

        # -------- STAY / LEAVE --------
        for zid in zone_ids:
            if zid in detected_now:
                zone_state[zid] = True
                last_seen_time[zid] = now
            else:
                if now - last_seen_time[zid] > LEAVE_CONFIRM_TIME:
                    zone_state[zid] = False

        # -------- TIME & ENERGY --------
        dt = now - last_update
        last_update = now
        for zid in zone_ids:
            if zone_state[zid]:
                zone_active_time[zid] += dt
                zone_energy[zid] += FAN_POWER_KW * dt / 3600
        total_energy = sum(zone_energy.values())

        # -------- RELAY CONTROL --------
        for zid in zone_ids:
            if zone_state[zid] != prev_zone_state[zid]:
                set_relay(zid, zone_state[zid])
                print(f"Zone {zid} -> {'FAN ON' if zone_state[zid] else 'FAN OFF'}")
        prev_zone_state = zone_state.copy()

        # -------- TOP BLYNK ROW --------
        active = [f"Zone{z}" for z in zone_ids if zone_state[z]]
        blynk_write(VPIN_ZONE_TEXT, ",".join(active) if active else "None")
        blynk_write(VPIN_ZONE_STATUS, "Fan ON" if active else "All Fans OFF")

        # -------- ENERGY CHART UPDATE --------
        if now - last_energy_push > 5:
            last_energy_push = now
            blynk_write(VPIN_ENERGY, round(total_energy, 3))

        # -------- LCD BELOW --------
        if now - last_lcd > 2:
            last_lcd = now
            runtime = max(1, now - program_start)
            select = blynk_read(VPIN_SELECT)

            if select == 0 and len(zone_ids) >= 2:
                lcd_write(
                    f"Z{zone_ids[0]} {zone_active_time[zone_ids[0]]/runtime*100:.1f}%",
                    f"Z{zone_ids[1]} {zone_active_time[zone_ids[1]]/runtime*100:.1f}%"
                )
            elif select == 1 and len(zone_ids) >= 2:
                lcd_write(
                    f"Z{zone_ids[0]} {int(zone_active_time[zone_ids[0]]/60)}m",
                    f"Z{zone_ids[1]} {int(zone_active_time[zone_ids[1]]/60)}m"
                )
            elif select == 2 and zone_ids:
                mz = max(zone_energy, key=zone_energy.get)
                lcd_write("Max Energy", f"Zone {mz} {zone_energy[mz]:.3f} kWh")

        time.sleep(0.05)

finally:
    GPIO.cleanup()
    cap.release()
