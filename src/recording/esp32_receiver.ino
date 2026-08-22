const int GSR_PIN = A0;  // D0 / GPIO0 on XIAO ESP32-C6

void setup() {
  Serial.begin(115200);
  pinMode(GSR_PIN, INPUT);
  analogReadResolution(12);  // Values from 0 to 4095
}

void loop() {
  uint32_t sumRaw = 0;
  uint32_t sumMillivolts = 0;

  // Average several samples to reduce noise
  for (int i = 0; i < 16; i++) {
    sumRaw += analogRead(GSR_PIN);
    sumMillivolts += analogReadMilliVolts(GSR_PIN);
    delay(5);
  }

  float raw = sumRaw / 16.0;
  float millivolts = sumMillivolts / 16.0;

  // Serial Plotter works best with labelled numeric columns
  Serial.print("GSR_raw:");
  Serial.print(raw);
  Serial.print("\tGSR_mV:");
  Serial.println(millivolts);

  delay(20);
}