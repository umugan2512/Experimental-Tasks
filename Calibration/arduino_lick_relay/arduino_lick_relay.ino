// Lick sensor relay for training01: OPB815WZ IR beam break -> Arduino -> Bpod BNC input.
//
// The Bpod's own lick port input never registered this sensor, so the Arduino reads the
// phototransistor and copies its state to a digital output wired to a Bpod BNC input. The Bpod
// timestamps every edge on its own clock (events BNC1High = lick in, BNC1Low = lick out);
// this loop adds ~10 us. No debouncing here, on purpose: filtering would delay the edges, and the
// analysis already merges sensor chatter from the raw edges (merge_lick_chatter()).
//
// Wiring (Uno/Nano, 5 V; check the colours against the OPB815WZ datasheet):
//   5V  -- 220 ohm -- Red   (LED anode)          ~15 mA through the IR LED
//   GND ------------- Black (LED cathode)
//   5V  -- 10 kohm -- White (collector) -- D2    pull-up; beam clear = LOW, beam broken = HIGH
//   GND ------------- Green (emitter)
//   D8  ------------- BNC centre pin  -> Bpod BNC In 1   (Bpod BNC inputs: optoisolated, 2.5-5 V)
//   GND ------------- BNC shield
//
// Check: Calibration/rig_check.py -> "Lick sensor / TTL monitor" shows which Bpod input fires.

const byte SENSOR_PIN = 2;
const byte OUT_PIN = 8;
const byte LED_PIN = LED_BUILTIN;   // on-board LED lit while the beam is broken

// false: output HIGH while the beam is broken (lick in = BNC1High). Set true only if the sensor
// is wired the other way round (beam broken reads LOW on D2).
const bool INVERT_OUTPUT = false;

void setup() {
  pinMode(SENSOR_PIN, INPUT);   // external 10 kohm pull-up
  pinMode(OUT_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  bool blocked = digitalRead(SENSOR_PIN) == HIGH;
  if (INVERT_OUTPUT) {
    blocked = !blocked;
  }
  digitalWrite(OUT_PIN, blocked ? HIGH : LOW);
  digitalWrite(LED_PIN, blocked ? HIGH : LOW);
}
